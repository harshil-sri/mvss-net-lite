from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

def apply_light_theme(slide, prs):
    # Set light background
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = RGBColor(255, 255, 255) # White

def create_title_banner(slide, title_text):
    # Create a blue banner at the top
    left = top = Inches(0)
    width = Inches(10) # Default slide width
    height = Inches(1.2)
    
    # Add a rectangle
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(0, 85, 164) # CDAC Blue
    shape.line.color.rgb = RGBColor(0, 85, 164)
    
    # Add title text inside the banner
    text_frame = shape.text_frame
    text_frame.word_wrap = True
    p = text_frame.paragraphs[0]
    p.text = title_text
    p.alignment = PP_ALIGN.LEFT
    p.font.name = 'Arial'
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = RGBColor(255, 255, 255)

def add_slide_custom(prs, title, content_list, is_title_slide=False):
    # Use blank layout to build custom UI
    blank_slide_layout = prs.slide_layouts[6] 
    slide = prs.slides.add_slide(blank_slide_layout)
    apply_light_theme(slide, prs)
    
    if is_title_slide:
        # Title Slide Design
        # Center big blue text
        left = Inches(1)
        top = Inches(2.5)
        width = Inches(8)
        height = Inches(2)
        txBox = slide.shapes.add_textbox(left, top, width, height)
        tf = txBox.text_frame
        
        p = tf.paragraphs[0]
        p.text = title
        p.alignment = PP_ALIGN.CENTER
        p.font.name = 'Arial'
        p.font.size = Pt(44)
        p.font.bold = True
        p.font.color.rgb = RGBColor(0, 85, 164) # CDAC Blue
        
        p2 = tf.add_paragraph()
        p2.text = content_list[0]
        p2.alignment = PP_ALIGN.CENTER
        p2.font.name = 'Arial'
        p2.font.size = Pt(24)
        p2.font.color.rgb = RGBColor(80, 80, 80)
        
        # Details at the bottom
        left = Inches(1)
        top = Inches(5.5)
        width = Inches(8)
        height = Inches(1.5)
        txBox_details = slide.shapes.add_textbox(left, top, width, height)
        tf_details = txBox_details.text_frame
        for line in content_list[1:]:
            if line:
                p_det = tf_details.add_paragraph()
                p_det.text = line
                p_det.alignment = PP_ALIGN.CENTER
                p_det.font.name = 'Arial'
                p_det.font.size = Pt(16)
                p_det.font.color.rgb = RGBColor(100, 100, 100)
                
    else:
        # Standard Slide Design
        create_title_banner(slide, title)
        
        # Add content box
        left = Inches(0.5)
        top = Inches(1.5)
        width = Inches(9)
        height = Inches(5.5)
        
        txBox = slide.shapes.add_textbox(left, top, width, height)
        tf = txBox.text_frame
        tf.word_wrap = True
        
        for idx, item in enumerate(content_list):
            if idx == 0:
                p = tf.paragraphs[0]
            else:
                p = tf.add_paragraph()
            p.text = "• " + item
            p.font.name = 'Arial'
            p.font.size = Pt(22)
            p.font.color.rgb = RGBColor(50, 50, 50)
            p.space_after = Pt(14)
            p.line_spacing = 1.2
            
    return slide

prs = Presentation()
# Ensure slide size is standard 4:3 (10x7.5 inches) or 16:9 (10x5.625 inches). Let's use 16:9
prs.slide_width = Inches(10)
prs.slide_height = Inches(5.625)

# Slide 1: Title
add_slide_custom(prs, "DocForge", [
    "Deep Learning based Document Forgery Detection",
    "",
    "Submitted By: Harshil Srivastav",
    "Roll No: 35817711624",
    "Semester: 5th, B.Tech (AIML)",
    "VIPS-TC | C-DAC, Noida",
    "Academic Year: 2026-27"
], is_title_slide=True)

# Slide 2: Introduction
add_slide_custom(prs, "1. Problem Statement", [
    "The rise of digital documents has led to a surge in sophisticated digital forgery.",
    "Traditional human verification is slow, subjective, and struggles with modern pixel-level manipulations (e.g., Photoshop, Generative AI).",
    "There is a critical need for automated systems to ensure integrity in Legal, Financial, and Administrative sectors.",
    "Manual inspection is unscalable in high-throughput environments."
])

# Slide 3: Objectives & Scope
add_slide_custom(prs, "2. Objectives & Scope", [
    "Design an automated, end-to-end deep learning pipeline for pixel-level forgery localization.",
    "Extract both high-level semantic features and low-level noise artifacts.",
    "Generalize across diverse forgery techniques (copy-move, splicing) and datasets.",
    "Provide a scalable solution for high-throughput, real-world deployment.",
    "Scope spans legal evidence verification, financial KYC processes, and insurance claim validation."
])

# Slide 4: Tools & Technologies
add_slide_custom(prs, "3. Tools & Technologies", [
    "Python & PyTorch: Core framework for model training, dynamic graphs, and tensor operations.",
    "OpenCV & NumPy: Image ingestion, geometric/photometric augmentations, and data manipulation.",
    "Deep Learning: Convolutional Neural Networks (CNNs) utilized for complex spatial feature extraction.",
    "Attention Mechanisms: Integration of Spatial and Channel attention to dynamically focus on manipulation artifacts rather than background textures."
])

# Slide 5: System Architecture
add_slide_custom(prs, "4. System Architecture", [
    "Data Ingestion & Preprocessing: Resizing, normalization, and robust morphological operations.",
    "Feature Extraction Backbone: Multi-layer CNN architecture extracting hierarchical image representations.",
    "Custom Attention-Based Fusion Module: Dynamically emphasizes structural anomalies and spatial discrepancies indicative of forgery.",
    "Prediction Head: Upsamples attention-weighted features to generate a pixel-wise probability map of manipulated regions."
])

# Slide 6: Implementation Details
add_slide_custom(prs, "5. Implementation Highlights", [
    "Dataset Balancing: Implemented WeightedRandomSampler to systematically handle imbalanced authentic vs. forged document distributions.",
    "Custom Loss Function: Developed 'CombinedLoss' integrating weighted Binary Cross Entropy (BCE) and Dice/Region Loss.",
    "Optimization: Trained with Batch Size 8, Learning Rate 1e-4, using pos_weight scaling for the minority positive class.",
    "Data Augmentation Pipeline: Extensive use of rotations, blur, flips, and photometric distortions to enhance model robustness."
])

# Slide 7: Evaluation & Results
add_slide_custom(prs, "6. Evaluation & Results", [
    "Standard Metrics Evaluated: Precision, Recall, F1-Score, and Intersection over Union (IoU).",
    "Performance: The CombinedLoss formulation successfully maximized IoU and F1-Score on the validation datasets.",
    "Visual Evidence: Predicted probability masks tightly correlate with ground truth, effectively isolating manipulations like spliced text and logos.",
    "Training Stability: Stable loss minimization observed across training epochs without severe overfitting."
])

# Slide 8: Conclusion & Future Scope
add_slide_custom(prs, "7. Conclusion & Future Scope", [
    "DocForge successfully automates the localization of subtle digital manipulations with high interpretability and accuracy.",
    "Future Work (Edge Deployment): Model quantization for deployment on smartphones and standalone scanners.",
    "Future Work (Dataset Expansion): Incorporating advanced GAN and Deepfake generated forgeries.",
    "Future Work (Multi-Modal): Extending the architecture for real-time video forgery detection and NLP-based semantic consistency checks."
])

output_path = "/mnt/c/Users/Harshil/Downloads/DocForge_Presentation_CDAC_Theme.pptx"
prs.save(output_path)
print(f"Presentation saved successfully to {output_path}")
