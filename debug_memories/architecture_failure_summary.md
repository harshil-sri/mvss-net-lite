# MVSS-Net Lite Architecture Failure Summary

## 1. The Core Problem
We are attempting to train/fine-tune a lightweight variant of MVSS-Net (MVSS-Net Lite) for forensic image forgery detection, specifically focusing on the **RTM (Real Text Manipulation)** dataset. 

Despite advanced debugging, the model has fundamentally failed to learn the task. Even when forced to **overfit a single batch of 8 images for 100 epochs**, it failed to correctly map the ground truth masks, outputting incorrect predictions on the exact images it trained on.

## 2. Debugging Timeline & Symptoms
1. **Initial Mode Collapse**: The model originally suffered from catastrophic mode collapse, predicting entirely black masks (Authentic) for everything. This was traced to extreme class imbalance (manipulated pixels make up < 1% of the document image).
2. **Loss Function Overhaul**: We replaced the naive BCE loss with a `CombinedLoss` (BCE with a heavy `pos_weight=50.0` + Dice Loss) to aggressively penalize false negatives.
3. **The "Text Detector" Symptom**: The new loss cured the mode collapse, but the model became hyper-sensitive. Instead of finding *forged* text, it began acting as a generic *text detector*, highlighting almost all text on the document. 
4. **Thresholding Fix**: We raised the inference probability threshold to `0.97` to slice off the false-positive text. This worked on a single local sample, but failed to generalize.
5. **The Overfit Failure (The Smoking Gun)**: To mathematically prove if the architecture was capable of learning the task, we wrote an `--overfit-batch` script that locks onto 8 specific images and trains on them for 100 epochs. **The model failed to overfit.** A mathematically sound deep learning architecture should be able to achieve near 0 loss and perfect pixel accuracy on a single mini-batch in 90 seconds. The fact that it cannot do this proves there is a fundamental flaw in the architecture or data flow.

## 3. Architectural Suspects (For the Higher Reasoning Model)
The higher reasoning model should investigate the following critical failure points:

### A. The Constrained Convolution (Noise Branch) is Blind
MVSS-Net utilizes a dual-branch architecture. The "noise branch" uses a Bayar constrained convolution (where the center weight is `-1` and the surrounding weights sum to `1`) designed to extract high-frequency noise variance and JPEG compression artifacts. 
- **Hypothesis**: The manipulations in the RTM dataset (synthesized text, digital splicing) might not leave traditional high-frequency camera noise or JPEG artifacts. If the text was spliced cleanly, the noise branch sees nothing. Because the network expects the noise branch to differentiate "real text" from "fake text" (while the RGB branch just sees "text"), a blind noise branch forces the network to just output the RGB branch's "text detector" signal.

### B. Capacity Bottlenecks in the "Lite" Variant
This is MVSS-Net *Lite*. If the backbone was aggressively scaled down (e.g., swapping ResNet50 for a heavily pruned ResNet18 or MobileNet, or drastically reducing the decoder channels), it may simply lack the parameter capacity to learn complex boundary inconsistencies, reducing it to learning low-level contrast features (text vs. background).

### C. Conflicting Gradients in CBAM Fusion
The two branches are fused using a CBAM (Convolutional Block Attention Module). If the noise branch is outputting garbage (due to point A), the spatial and channel attention mechanisms in CBAM might be collapsing or fighting the RGB gradients, preventing the network from descending the loss landscape even on a single batch.

### D. Over-Aggressive Data Augmentation
During the overfit test, the batch is grabbed from the dataloader after `is_train=True` augmentations are applied. The pipeline includes random color jittering, Gaussian blur, and scaling. 
- **Hypothesis**: Gaussian blur and color jitter physically destroy high-frequency forensic noise. We might be literally erasing the evidence of forgery before the model even sees it.

## 4. Suggested Next Steps
1. **Isolate the Branches**: Modify `train.py` or the model architecture to run *only* the RGB branch, completely disabling the constrained noise branch and CBAM fusion. Run the overfit test again. If it succeeds, the noise branch/fusion is the culprit.
2. **Disable Augmentations**: Strip all Gaussian blur, rotation, and color jitter from the dataloader. Run the overfit test on clean, unadulterated pixels to see if the forensic traces survive.
3. **Verify the CBAM gradients**: Check if the attention weights in the CBAM module are saturating to 0 or 1, effectively killing the gradient flow to the encoder.
