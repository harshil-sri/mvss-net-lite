from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

def apply_dark_theme(slide):
    # Set dark background
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = RGBColor(30, 30, 30)

def set_font_color(shape, r, g, b, bold=False):
    if not shape.has_text_frame:
        return
    for paragraph in shape.text_frame.paragraphs:
        for run in paragraph.runs:
            run.font.color.rgb = RGBColor(r, g, b)
            if bold:
                run.font.bold = True

def add_slide(prs, title, content_list, slide_type=1):
    # slide_type 1 is bullet points, 0 is title slide
    slide_layout = prs.slide_layouts[slide_type]
    slide = prs.slides.add_slide(slide_layout)
    apply_dark_theme(slide)
    
    title_shape = slide.shapes.title
    title_shape.text = title
    set_font_color(title_shape, 240, 240, 240, bold=True)
    
    if slide_type == 1 and len(slide.placeholders) > 1:
        body_shape = slide.placeholders[1]
        tf = body_shape.text_frame
        tf.text = content_list[0]
        for item in content_list[1:]:
            p = tf.add_paragraph()
            p.text = item
        set_font_color(body_shape, 200, 200, 200)
    elif slide_type == 0 and len(slide.placeholders) > 1:
        subtitle_shape = slide.placeholders[1]
        tf = subtitle_shape.text_frame
        tf.text = "\n".join(content_list)
        set_font_color(subtitle_shape, 180, 180, 180)
        
    return slide

prs = Presentation()

# Slide 1: Title
add_slide(prs, "DocForge", [
    "Deep Learning based Document Forgery Detection",
    "",
    "Submitted By: Harshil Srivastav",
    "Roll No: 35817711624",
    "Semester: 5th, B.Tech (AIML)",
    "VIPS-TC | C-DAC, Noida",
    "Academic Year: 2026-27"
], slide_type=0)

# Slide 2: Introduction
add_slide(prs, "1. Problem Statement", [
    "The rise of digital documents has led to a surge in sophisticated digital forgery.",
    "Traditional human verification is slow, subjective, and struggles with modern pixel-level manipulations (e.g., Photoshop, Generative AI).",
    "There is a critical need for automated systems to ensure integrity in Legal, Financial, and Administrative sectors.",
    "Manual inspection is unscalable in high-throughput environments."
])

# Slide 3: Objectives & Scope
add_slide(prs, "2. Objectives & Scope", [
    "Design an automated, end-to-end deep learning pipeline for pixel-level forgery localization.",
    "Extract both high-level semantic features and low-level noise artifacts.",
    "Generalize across diverse forgery techniques (copy-move, splicing) and datasets.",
    "Provide a scalable solution for high-throughput, real-world deployment.",
    "Scope spans legal evidence verification, financial KYC processes, and insurance claim validation."
])

# Slide 4: Tools & Technologies
add_slide(prs, "3. Tools & Technologies", [
    "Python & PyTorch: Core framework for model training, dynamic graphs, and tensor operations.",
    "OpenCV & NumPy: Image ingestion, geometric/photometric augmentations, and data manipulation.",
    "Deep Learning: Convolutional Neural Networks (CNNs) utilized for complex spatial feature extraction.",
    "Attention Mechanisms: Integration of Spatial and Channel attention to dynamically focus on manipulation artifacts rather than background textures."
])

# Slide 5: System Architecture
add_slide(prs, "4. System Architecture", [
    "Data Ingestion & Preprocessing: Resizing, normalization, and robust morphological operations.",
    "Feature Extraction Backbone: Multi-layer CNN architecture extracting hierarchical image representations.",
    "Custom Attention-Based Fusion Module: Dynamically emphasizes structural anomalies and spatial discrepancies indicative of forgery.",
    "Prediction Head: Upsamples attention-weighted features to generate a pixel-wise probability map of manipulated regions."
])

# Slide 6: Implementation Details
add_slide(prs, "5. Implementation Highlights", [
    "Dataset Balancing: Implemented WeightedRandomSampler to systematically handle imbalanced authentic vs. forged document distributions.",
    "Custom Loss Function: Developed 'CombinedLoss' integrating weighted Binary Cross Entropy (BCE) and Dice/Region Loss.",
    "Optimization: Trained with Batch Size 8, Learning Rate 1e-4, using pos_weight scaling for the minority positive class.",
    "Data Augmentation Pipeline: Extensive use of rotations, blur, flips, and photometric distortions to enhance model robustness."
])

# Slide 7: Evaluation & Results
add_slide(prs, "6. Evaluation & Results", [
    "Standard Metrics Evaluated: Precision, Recall, F1-Score, and Intersection over Union (IoU).",
    "Performance: The CombinedLoss formulation successfully maximized IoU and F1-Score on the validation datasets.",
    "Visual Evidence: Predicted probability masks tightly correlate with ground truth, effectively isolating manipulations like spliced text and logos.",
    "Training Stability: Stable loss minimization observed across training epochs without severe overfitting."
])

# Slide 8: Conclusion & Future Scope
add_slide(prs, "7. Conclusion & Future Scope", [
    "DocForge successfully automates the localization of subtle digital manipulations with high interpretability and accuracy.",
    "Future Work (Edge Deployment): Model quantization for deployment on smartphones and standalone scanners.",
    "Future Work (Dataset Expansion): Incorporating advanced GAN and Deepfake generated forgeries.",
    "Future Work (Multi-Modal): Extending the architecture for real-time video forgery detection and NLP-based semantic consistency checks."
])

output_path = "/mnt/c/Users/Harshil/Downloads/DocForge_Presentation.pptx"
prs.save(output_path)
print(f"Presentation saved successfully to {output_path}")
