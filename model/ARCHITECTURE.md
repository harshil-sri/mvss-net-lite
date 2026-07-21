# MVSS-Net Lite Architecture

![Architecture Diagram](../reports/architecture_diagram.png)

This document breaks down every file in the `model/` directory of the MVSS-Net Lite project. It is designed to be read end-to-end as a continuous pipeline, explaining not just *what* the code does, but *why* it is designed this way.

---

## 1. `model/constrained_conv.py`

**What problem this file solves**
When detecting forged images, the actual content of the image (like a dog or a stop sign) is usually a distraction. What we actually care about is invisible noise patterns—specifically, whether a pasted object has a different compression footprint or microscopic static pattern than the rest of the image. This file creates a special convolution layer that ignores the picture's visual content and strictly highlights these microscopic noise differences.

**A non-technical analogy**
Imagine a teacher trying to figure out if a student copy-pasted a paragraph into their essay. Instead of reading the *meaning* of the words (the visual content), the teacher pulls out a magnifying glass and looks at the *ink*—checking if the font weight or the spacing drastically changes from one sentence to its immediate neighbors. This layer looks purely at the "ink."

**Code walkthrough**
```python
    def normalize_weights(self):
        with torch.no_grad():
            c_h, c_w = self.kernel_size[0] // 2, self.kernel_size[1] // 2
            
            # Temporarily zero out the center weights
            self.weight.data[:, :, c_h, c_w] = 0.0
            
            # Sum over spatial dimensions
            sums = self.weight.data.sum(dim=(2, 3), keepdim=True)
            sums[sums == 0] = 1.0
            
            # Normalize so surrounding weights sum to +1
            self.weight.data /= sums
            
            # Fix center weights to -1
            self.weight.data[:, :, c_h, c_w] = -1.0
```
This is the Bayar constrained convolution logic. By forcing the center weight of the filter to be exactly `-1` and the sum of all surrounding weights to be exactly `+1`, the filter is mathematically forced to act as a high-pass residual filter. It subtracts the center pixel's value from the average of its neighbors. If a pixel is perfectly identical to its neighbors (a flat color), the result is `0` (the content is suppressed). If a pixel deviates from its neighbors (noise), it gets highlighted. 

**Shape trace**
Assuming an input image of 256x256:
`[B, 3, 256, 256]` -> ConstrainedConv2d (padding preserves size) -> `[B, 3, 256, 256]` (Noise Residuals)

**How it connects to the next file**
This file acts as the absolute front door for the "Noise Branch." The raw RGB image goes into this layer, gets stripped of its visual content to reveal pure noise residuals, and is then handed off to the `model/backbone.py` to be deeply analyzed.

---

## 2. `model/backbone.py`

**What problem this file solves**
Once we have our raw RGB image and our extracted noise residuals, we need to find deep, abstract patterns in them. This file spins up two massive feature extractors based on the ResNet-34 architecture. It passes the noise through one extractor and the RGB through the other, pulling out patterns at four different resolutions (stages).

**A non-technical analogy**
Imagine an assembly line with two parallel conveyor belts. On the top belt, workers are inspecting a car's engine purely for weird clicking sounds (the Noise Branch). On the bottom belt, workers are inspecting the car's paint job and exterior (the RGB Branch). Both belts run completely independently, and the workers never talk to each other, but at four different checkpoints along the line, they both hand their clipboards over to a manager (the fusion module). 

**Code walkthrough**
```python
        # Cross-branch fusion at each stage
        f1 = self.fuse1(n1, e1)
        f2 = self.fuse2(n2, e2)
        f3 = self.fuse3(n3, e3)
        f4 = self.fuse4(n4, e4)
        
        return f1, f2, f3, f4
```
Here we see the backbone generating its final outputs. Notice that `n1` (noise stage 1) and `e1` (edge/RGB stage 1) are fed into `self.fuse1`, but the result `f1` *does not go back into the next stage of the backbone*. The two ResNets run entirely parallel and separate. 
*Design Tradeoff:* Why two separate ResNets instead of one shared one? Because the inputs are fundamentally different (raw colors vs. microscopic noise). If they shared weights, the network would get confused trying to find semantic shapes in static noise. Keeping them separate allows each branch to become highly specialized.

**Shape trace**
(Assuming 256x256 input)
*   **Stem (Conv1+Pool):** `[B, 3, 256, 256]` -> `[B, 64, 64, 64]`
*   **Layer 1 (`n1`/`e1`):** `[B, 64, 64, 64]` -> `[B, 64, 64, 64]`
*   **Layer 2 (`n2`/`e2`):** `[B, 64, 64, 64]` -> `[B, 128, 32, 32]`
*   **Layer 3 (`n3`/`e3`):** `[B, 128, 32, 32]` -> `[B, 256, 16, 16]`
*   **Layer 4 (`n4`/`e4`):** `[B, 256, 16, 16]` -> `[B, 512, 8, 8]`

**How it connects to the next file**
The backbone doesn't make any final decisions. Instead, it delegates the blending of these two parallel streams to the `model/fusion.py` modules at each of the four stages, outputting `f1`, `f2`, `f3`, and `f4`.

---

## 3. `model/fusion.py`

**What problem this file solves**
We have noise features and RGB features, but they aren't equally important at all times. If a forgery is a sloppy copy-paste, the RGB branch might easily spot mismatched lighting. If it's a high-quality deepfake, the RGB branch might see nothing, but the Noise branch will scream. This file uses attention mechanisms (CBAM) to dynamically decide which branch to listen to at any given pixel before blending them together.

**A non-technical analogy**
Imagine a sound engineer mixing a live concert. Instead of just leaving the drummer and the guitarist at the exact same volume (element-wise addition), the engineer actively listens. When the guitarist plays a solo, the engineer turns up the guitar slider and turns down the drums (attention). This file is the sound engineer, dynamically tweaking the volumes of the Noise and RGB signals before mixing them into one master track.

**Code walkthrough**
```python
    def forward(self, noise_feat, edge_feat):
        n = self.channel_attn_noise(noise_feat)
        n = self.spatial_attn_noise(n)

        e = self.channel_attn_edge(edge_feat)
        e = self.spatial_attn_edge(e)

        combined = torch.cat([n, e], dim=1)   
        fused = self.fuse_conv(combined)      
```
First, channel and spatial attention are applied independently to both streams. Then, they are concatenated (`torch.cat`) and squashed back down using a 1x1 convolution (`fuse_conv`). 
*Design Tradeoff:* Why concatenate and convolve instead of simply adding them together (`n + e`)? Addition forces a rigid 50/50 blend. Concatenation gives the 1x1 convolution layer the freedom to mathematically learn exactly how much of channel X from the noise branch should mix with channel Y from the RGB branch.

**Shape trace**
Looking at Stage 4 as an example:
*   Inputs: `noise_feat [B, 512, 8, 8]`, `edge_feat [B, 512, 8, 8]`
*   Concat: `[B, 1024, 8, 8]`
*   1x1 Conv: `[B, 512, 8, 8]` (This is `f4`)

**How it connects to the next file**
This file produces the four fused "master tracks" (`f1` through `f4`). These fully blended feature maps are then handed directly to the master blueprint in `model/network.py` to be decoded into actual images.

---

## 4. `model/network.py`

**What problem this file solves**
This is the central hub. It wires the entire architecture together. It instantiates the backbone and fusion layers, and most importantly, it defines the "Decoder." The decoder's job is to take the tiny, deeply compressed features from the backbone and scale them back up into full-resolution, pixel-perfect images (the segmentation mask and edge map).

**A non-technical analogy**
Imagine a general contractor building a house. They don't pour the concrete themselves; they just hire the foundation crew (backbone), the framing crew (fusion), and the painters (decoder), and they ensure materials flow smoothly from one team to the next. The decoder itself acts like a police sketch artist, taking abstract, vague descriptions from detectives (the deeply pooled layers) and combining them with sharp photo references (the skip connections) to draw a high-resolution, accurate face.

**Code walkthrough**
```python
        # Decoder with progressive upsampling and skip connection concatenation
        x = F.interpolate(f4, size=f3.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, f3], dim=1)
        x = self.up_conv3(x)
```
This is a classic U-Net style decoder block. It takes the tiny, highly abstract `f4` layer and scales it up spatially using interpolation. But interpolation is blurry! To get the sharpness back, it concatenates `x` with `f3` (a skip connection from an earlier, higher-resolution stage of the backbone). The convolution `up_conv3` then blends this deep semantic understanding with the sharp spatial details.

**Shape trace**
Starting from the deepest feature (`f4`):
*   `f4 [B, 512, 8, 8]` -> Upsample -> `[B, 512, 16, 16]`
*   Concat with `f3 [B, 256, 16, 16]` -> `[B, 768, 16, 16]`
*   `up_conv3` -> `[B, 256, 16, 16]`
*   ... repeats until ...
*   Final Output Heads -> `[B, 1, 256, 256]` (one for mask, one for edges)

**How it connects to the next file**
This file defines the literal PyTorch `nn.Module` that represents our AI. It is imported by `model/train.py`, which brings it to life by feeding it data and punishing its mistakes via a loss function.

---

## 5. `model/train.py`

**What problem this file solves**
An AI model is just a block of random math until it learns. This file runs the actual training loop. It pulls images from the dataset, passes them through the network, compares the network's guesses against the ground-truth answers, and tweaks the network's internal math (weights) so it guesses better next time.

**A non-technical analogy**
Imagine a coach drilling a tennis player. The coach serves a ball (dataloader), watches where the player hits it (forward pass), calculates how far out of bounds the ball landed (loss function), and physically adjusts the player's grip and stance (optimizer step) before throwing the next ball.

**Code walkthrough**
```python
            # Compute losses
            loss_seg = seg_criterion(pred_seg, masks)
            loss_edge = edge_criterion(pred_edge, edges)
            
            loss_total = (SEG_LOSS_WEIGHT * loss_seg) + (EDGE_LOSS_WEIGHT * loss_edge)
            
            # Backward pass & step
            loss_total.backward()
            optimizer.step()
            
            # Normalize the constrained conv layer weights!
            model.backbone.noise_extractor.normalize_weights()
```
The network predicts both a segmentation mask and an edge map, and we calculate Binary Cross Entropy (BCE) for both. We sum these losses together into a `loss_total`, allowing the network to learn both tasks simultaneously (Multi-Task Learning). The critical step is `model.backbone.noise_extractor.normalize_weights()`. Because the optimizer blindly updates weights to minimize loss, it will destroy the strict mathematical constraints of the Bayar Conv layer. We must manually reset those constraints immediately after every single optimizer step.

**Shape trace**
*   `imgs`: `[B, 3, 256, 256]`
*   `masks` & `edges` (Ground Truth): `[B, 1, 256, 256]`
*   `pred_seg` & `pred_edge` (Network Output): `[B, 1, 256, 256]`
*   `loss_total`: A single scalar value (e.g., `0.4532`)

**How it connects to the next file**
This script produces the final, trained model `.pt` weights, which are saved to `model/checkpoints/` to be used in production or evaluation scripts.

---

## Common Questions a Mentor Might Ask

**Q: Why use a Constrained Convolution instead of letting the network just learn its own standard convolution filter?**
A: If you use a standard convolution, the network gets lazy. It will immediately latch onto the easiest visual features (like text color, background gradients, lines) to try and solve the problem, completely ignoring the subtle, microscopic noise patterns. Forcing a constrained convolution acts as a hard mathematical block that destroys visual content, physically forcing that branch of the network to only look at noise residuals.

**Q: Why do we have two separate ResNet-34 backbones instead of sharing weights between them?**
A: Shared weights are used when the inputs are highly similar (like left/right stereo cameras). Here, our inputs are totally different: one branch sees standard colorful pictures, and the other sees invisible high-frequency static. If they shared weights, the filters would suffer catastrophic interference trying to parse semantic objects and pure noise simultaneously. 

**Q: Why do we fuse features at every stage instead of just once at the very end?**
A: Forgery artifacts exist at multiple scales. A blurred edge might be visible at a 256x256 resolution, but entirely pooled away and lost by the time the ResNet reaches its 8x8 stage. Conversely, a lighting mismatch is obvious at 8x8 but too large to see at 256x256. Fusing at every stage ensures the decoder has access to anomalies of all sizes.

**Q: What happens if you remove the Edge Map head entirely?**
A: The model will still output a segmentation mask, but the borders of the predicted mask will likely become blobby, rounded, and imprecise. The edge head acts as an "auxiliary task." By forcing the network to also explicitly predict the razor-thin boundary of the forgery, it mathematically forces the shared backbone features to preserve high-frequency spatial awareness, resulting in much sharper primary segmentation masks.

**Q: Why use `BCEWithLogitsLoss` instead of applying a Sigmoid inside the network and using standard `BCELoss`?**
A: Numerical stability. If you apply a Sigmoid in the network, extreme values are pushed to exactly `0` or `1`, which causes gradients to vanish (become exactly zero) due to floating-point rounding. `BCEWithLogitsLoss` combines the Sigmoid and the BCE math into a single robust operation under the hood, preventing catastrophic vanishing gradients during training.

**Q: How is our "MVSS-Net Lite" architecture better (or different) than the original MVSS-Net?**
A: The original MVSS-Net is a heavyweight architecture built for absolute maximum benchmark accuracy. It often uses a massive ResNet-50 or SegFormer backbone and extremely complex multi-scale cross-attention rules that consume huge amounts of VRAM. Our "Lite" version strategically scales this down to dual ResNet-34s, making it fast enough to train on standard consumer GPUs and run inference in real-time. Additionally, we completely replaced the original model's thick, blocky morphological edge ground-truth approach with a razor-thin Canny edge extraction. We also streamlined the fusion mechanism into a single, clean CBAM block per stage instead of a tangled web of cross-attention, drastically reducing the parameter count while completely preserving the core dual-stream (noise + RGB) philosophy.
