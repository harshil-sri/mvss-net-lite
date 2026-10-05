import os
import re
from docx import Document
from docx.shared import Pt, Cm, Mm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

def set_page_margins(section):
    section.page_width = Mm(210)
    section.page_height = Mm(297)
    section.left_margin = Cm(3.5)
    section.top_margin = Cm(2.5)
    section.right_margin = Cm(1.25)
    section.bottom_margin = Cm(1.25)

def add_page_number(run):
    fldChar1 = OxmlElement('w:fldChar')
    fldChar1.set(qn('w:fldCharType'), 'begin')
    instrText = OxmlElement('w:instrText')
    instrText.set(qn('xml:space'), 'preserve')
    instrText.text = "PAGE"
    fldChar2 = OxmlElement('w:fldChar')
    fldChar2.set(qn('w:fldCharType'), 'separate')
    fldChar3 = OxmlElement('w:fldChar')
    fldChar3.set(qn('w:fldCharType'), 'end')
    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)
    run._r.append(fldChar3)

def set_page_number_format(section, fmt='lowerRoman'):
    sectPr = section._sectPr
    pgNumType = OxmlElement('w:pgNumType')
    pgNumType.set(qn('w:fmt'), fmt)
    pgNumType.set(qn('w:start'), '1')
    sectPr.append(pgNumType)

def add_toc(paragraph):
    run = paragraph.add_run()
    fldChar1 = OxmlElement('w:fldChar')
    fldChar1.set(qn('w:fldCharType'), 'begin')
    instrText = OxmlElement('w:instrText')
    instrText.set(qn('xml:space'), 'preserve')
    instrText.text = 'TOC \\o "1-3" \\h \\z \\u'
    fldChar2 = OxmlElement('w:fldChar')
    fldChar2.set(qn('w:fldCharType'), 'separate')
    fldChar3 = OxmlElement('w:fldChar')
    fldChar3.set(qn('w:fldCharType'), 'end')
    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)
    run._r.append(fldChar3)

doc = Document()

style = doc.styles['Normal']
font = style.font
font.name = 'Times New Roman'
font.size = Pt(12)
style.paragraph_format.line_spacing = 1.5

def add_paragraph(text, size=12, bold=False, align=WD_ALIGN_PARAGRAPH.LEFT, spacing=1.5):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.line_spacing = spacing
    if text:
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = RGBColor(0,0,0)
    return p

def style_heading(h, size):
    for run in h.runs:
        run.font.name = 'Times New Roman'
        run.font.size = Pt(size)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0,0,0)

# Section 1: Front Matter
section_1 = doc.sections[0]
set_page_margins(section_1)
set_page_number_format(section_1, 'lowerRoman')

footer_1 = section_1.footer
footer_1.is_linked_to_previous = False
footer_p = footer_1.paragraphs[0]
footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
add_page_number(footer_p.add_run())

# Cover Page
for _ in range(5): doc.add_paragraph()
add_paragraph("SUMMER INTERNSHIP REPORT", size=24, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()
add_paragraph("DOCFORGE", size=16, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()
doc.add_paragraph()
add_paragraph("Submitted in partial fulfillment of the Requirements for the award of Degree of Bachelor of Technology in Artificial Intelligence and Machine Learning (AI/ML)", size=12, align=WD_ALIGN_PARAGRAPH.CENTER)
for _ in range(3): doc.add_paragraph()
add_paragraph("Submitted By", size=14, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
add_paragraph("Name: Harshil Srivastav\nUniversity Roll No.: 35817711624\nSemester: 5th\nAcademic Year: 2026-27", size=14, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)

doc.add_page_break()

# Declaration
for _ in range(2): doc.add_paragraph()
add_paragraph("DECLARATION", size=16, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()
add_paragraph('I hereby declare that the Internship Report entitled "DocForge" is an authentic record of my own work as requirements of Internship during the period from 15.06.2026 to 31.07.2026 for the award of degree of B.Tech. (AIML) School of Engineering and Technology, VIPS-TC.', size=12, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
doc.add_paragraph()
doc.add_paragraph()
add_paragraph("(Signature of student)\nHarshil Srivastav\n35817711624\nDate: 25 September 2026", size=12)

doc.add_page_break()

# Certificate
for _ in range(2): doc.add_paragraph()
add_paragraph("CERTIFICATE", size=16, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()
add_paragraph("[INSERT C-DAC CERTIFICATE IMAGE HERE]", size=12, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()
add_paragraph("This is to certify that Mr. Harshil Srivastav has completed Internship during the period from 15.06.2026 to 31.07.2026 in our Organization / Industry as a Partial Fulfillment of Degree of Bachelor of Technology in AIML. He was trained in the field of Artificial Intelligence and Machine Learning / Document Forgery Detection.", size=12, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
doc.add_paragraph()
doc.add_paragraph()
add_paragraph("Signature & Seal of Training Manager\nMr. Tushar Kanti Patnaik, Scientist F", size=12)

doc.add_page_break()

# Acknowledgement
for _ in range(2): doc.add_paragraph()
add_paragraph("ACKNOWLEDGEMENT", size=14, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()
add_paragraph("I would like to express my profound gratitude to the Centre for Development of Advanced Computing (C-DAC), Noida, for providing me with the invaluable opportunity to pursue my summer internship. This experience has significantly enriched my practical understanding of machine learning and computer vision.", size=12, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
add_paragraph("I am deeply indebted to my project guide, Mr. Tushar Kanti Patnaik, Scientist F, for his continuous guidance, encouragement, and technical mentorship throughout the development of the \"DocForge\" project. His insights were instrumental in navigating the complexities of deep learning pipelines.", size=12, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
add_paragraph("I also extend my sincere thanks to the faculty and administration of Vivekananda Institute of Professional Studies - Technical Campus (VIPS-TC), Pitampura, for equipping me with the foundational knowledge required to undertake this training and for their continuous academic support. Finally, I would like to thank my family and peers for their unwavering encouragement throughout this endeavor.", size=12, align=WD_ALIGN_PARAGRAPH.JUSTIFY)

doc.add_page_break()

# About Company
for _ in range(2): doc.add_paragraph()
add_paragraph("ABOUT COMPANY", size=14, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()
add_paragraph("The Centre for Development of Advanced Computing (C-DAC) is the premier research and development organization of the Ministry of Electronics and Information Technology (MeitY), Government of India. Established to advance the country's IT and electronics sector, C-DAC engages in cutting-edge research in areas such as High-Performance Computing (HPC), Artificial Intelligence, Cyber Security, and Software Technologies. The Noida facility, located in the Institutional Area of Sector 62, actively drives technological innovation, education, and specialized training programs designed to bridge the gap between academic learning and industry demands.", size=12, align=WD_ALIGN_PARAGRAPH.JUSTIFY)

doc.add_page_break()

# TOC
for _ in range(2): doc.add_paragraph()
add_paragraph("TABLE OF CONTENTS", size=14, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()
p_toc = doc.add_paragraph()
add_toc(p_toc)
add_paragraph("(To update this TOC, right click here in Word and select 'Update Field')", size=10, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER)

# Section 2: Main Chapters
section_2 = doc.add_section(WD_SECTION.NEW_PAGE)
set_page_margins(section_2)
set_page_number_format(section_2, 'decimal')
footer_2 = section_2.footer
footer_2.is_linked_to_previous = False
footer_p2 = footer_2.paragraphs[0]
footer_p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
add_page_number(footer_p2.add_run())


CONTENT = [
    ("# CHAPTER 1: INTRODUCTION TO PROJECT", 1, 14),
    ("## 1.1 Problem Statement", 2, 12),
    ("The advent of the digital era has ushered in an unprecedented reliance on digital documents across nearly every facet of modern society. From legal contracts, identity documents, and financial records to academic transcripts and medical histories, the digitization of information has vastly improved the efficiency of storage, retrieval, and transmission. However, this convenience is intrinsically accompanied by a significant vulnerability: the ease with which digital documents can be manipulated or forged.", 0, 12),
    ("Historically, document forgery involved physical alterations—erasing ink, forging signatures by hand, or carefully pasting elements from one document to another. The verification of such documents, while challenging, relied on the physical properties of the paper, ink analysis, and the keen eye of trained forensic document examiners. In contrast, modern digital forgery leverages sophisticated image editing software like Adobe Photoshop, GIMP, and increasingly, generative artificial intelligence models. These tools democratize the ability to alter digital images, allowing even novice users to seamlessly perform copy-move, splicing, and retouching operations without leaving visible traces to the naked eye.", 0, 12),
    ("The limitations of human verification in detecting these sophisticated digital manipulations are profound. Human visual perception is generally ill-equipped to identify subtle statistical anomalies, pixel-level discrepancies, or complex lighting inconsistencies introduced during the forgery process. Manual inspection is not only highly subjective and error-prone but also extremely time-consuming and unscalable in environments where thousands of documents are processed daily, such as immigration checkpoints, insurance claim processing centers, or large-scale financial institutions.", 0, 12),
    ("Consequently, there is an urgent and critical need for automated, intelligent systems capable of detecting document manipulations with high accuracy and reliability. This project, DocForge, addresses this critical gap by proposing a robust, deep-learning-based manipulation detection system designed to analyze digital documents and localize manipulated regions, thereby restoring trust in digital documentation.", 0, 12),
    
    ("## 1.2 Objectives", 2, 12),
    ("The primary objective of the DocForge project is to design, develop, and evaluate a robust, state-of-the-art deep learning system capable of automatically detecting and localizing digital manipulations in scanned and digital-born documents. Specifically, the objectives are delineated as follows:", 0, 12),
    ("1. To engineer an automated detection pipeline: To transition away from manual, subjective verification processes by developing an automated pipeline that can ingest digital documents and output a pixel-level localization map indicating forged regions.", 0, 12),
    ("2. To design a robust feature extraction framework: To leverage advanced Convolutional Neural Networks (CNNs) capable of extracting both high-level semantic features and low-level noise artifacts, which are essential for identifying discrepancies introduced by forgery operations such as copy-move and splicing.", 0, 12),
    ("3. To implement an effective learning strategy: To train the system on diverse and complex datasets comprising both authentic and manipulated documents, ensuring the model generalizes well to unseen forgery techniques.", 0, 12),
    ("4. To provide a scalable and efficient solution: To ensure that the resulting system, while highly accurate, maintains computational efficiency suitable for deployment in real-world, high-throughput environments.", 0, 12),
    
    ("## 1.3 Scope", 2, 12),
    ("The scope of the DocForge project is expansive, given the ubiquitous nature of digital documents. The real-world applications of a robust document forgery detection system span multiple critical sectors, including the Legal Sector (integrity of digital evidence), Financial Sector (bank statements, identity cards), Administrative Sectors (passports, tax records), and Insurance Industry (fraudulent claims).", 0, 12),
    ("While the primary focus of this phase is on static image-based document forgery detection, the foundational architecture and methodologies developed in DocForge establish a critical baseline that can be extended in future iterations to encompass more complex media, such as multi-page PDFs and video-based identity verification.", 0, 12),
    
    ("# CHAPTER 2: TOOLS & TECHNOLOGY USED", 1, 14),
    ("## 2.1 Deep Learning Fundamentals", 2, 12),
    ("The foundation of the DocForge system lies in Deep Learning, a subfield of Machine Learning inspired by the structure and function of the human brain's neural networks. Deep learning models, specifically deep artificial neural networks, have revolutionized the field of Computer Vision by achieving unprecedented performance in tasks such as image classification, object detection, and semantic segmentation.", 0, 12),
    ("Deep learning models are built using artificial neural networks that have multiple layers—hence the term \"deep\". These layers consist of nodes, or artificial neurons, which process input data, apply mathematical transformations, and pass the results to the subsequent layer. The depth of these networks enables them to automatically learn complex, hierarchical feature representations directly from raw data, bypassing the need for tedious manual feature engineering that bottlenecked traditional computer vision approaches.", 0, 12),
    
    ("### 2.1.1 History and Evolution of Neural Networks", 3, 12),
    ("The conceptual framework for artificial neural networks dates back to the 1940s with the McCulloch-Pitts neuron model, which provided a simplified mathematical model of biological neurons. In the late 1950s, Frank Rosenblatt introduced the Perceptron, the first algorithmically described neural network capable of learning simple binary classification tasks. However, the Perceptron was fundamentally limited to linearly separable problems, as famously highlighted by Minsky and Papert in 1969.", 0, 12),
    ("It was not until the widespread popularization of the backpropagation algorithm in the 1980s that training multi-layer perceptrons (MLPs) became computationally feasible. Backpropagation allowed errors to be propagated backwards through the network layers, enabling the adjustment of internal hidden weights. Despite this breakthrough, the field experienced periods of stagnation known as \"AI winters\" due to severe computational limitations and the lack of sufficiently large annotated datasets.", 0, 12),
    ("The modern renaissance of deep learning was catalyzed in the early 2010s by three converging factors: the exponential increase in computational power provided by Graphics Processing Units (GPUs), the creation of massive open-source datasets like ImageNet, and the development of novel activation functions (like ReLU) that mitigated the vanishing gradient problem. The landmark victory of AlexNet in the 2012 ImageNet Large Scale Visual Recognition Challenge (ILSVRC) firmly established deep learning as the dominant paradigm in artificial intelligence.", 0, 12),
    
    ("### 2.1.2 Convolutional Neural Networks (CNNs)", 3, 12),
    ("Convolutional Neural Networks (CNNs) are a highly specialized class of deep neural networks designed explicitly for processing data with a known grid-like topology, such as 2D images. Unlike standard fully connected networks where every neuron is connected to every neuron in the adjacent layer, CNNs leverage the spatial correlation inherent in image data through three fundamental architectural concepts: local receptive fields, shared weights, and spatial subsampling.", 0, 12),
    ("1. Convolutional Layers: The core building block of a CNN. These layers apply a set of learnable filters (or kernels) across the input image. Each filter performs a mathematical operation called convolution, sliding across the spatial dimensions of the image and computing the dot product between the filter weights and the local input region. This process produces 2D activation maps, which represent the spatial presence of specific visual features, such as edges, textures, or more complex patterns. The use of shared weights drastically reduces the total number of parameters compared to fully connected networks, making CNNs highly computationally efficient and less prone to statistical overfitting.", 0, 12),
    ("2. Pooling Layers: Following convolutional operations, pooling layers are frequently employed to progressively reduce the spatial dimensions of the feature maps. Max pooling, the most prevalent variant, slides a window over the input and outputs the maximum value within that window. This subsampling introduces critical translation invariance, making the network robust to slight shifts, rotations, and distortions in the input image, while concurrently reducing computational complexity and memory requirements for subsequent layers.", 0, 12),
    ("3. Fully Connected Layers: Typically located at the terminal end of the network architecture, fully connected layers flatten the output of the preceding convolutional and pooling layers into a 1D vector and utilize these high-level aggregated features to perform the final classification or regression task.", 0, 12),
    
    ("### 2.1.3 Activation Functions and Optimization", 3, 12),
    ("Activation functions introduce essential non-linearity into the network, enabling it to learn and model complex, highly non-linear mappings between high-dimensional inputs and outputs. Without non-linear activation functions, a deep neural network, regardless of its depth, would mathematically collapse into a single linear transformation, severely limiting its representational capacity.", 0, 12),
    ("The Rectified Linear Unit (ReLU), defined mathematically as f(x) = max(0, x), is the most widely utilized activation function in modern CNNs. It is highly computationally efficient and effectively mitigates the vanishing gradient problem—a phenomenon where gradients become infinitesimally small during backpropagation, halting the learning process in deep networks.", 0, 12),
    ("Training a deep neural network involves finding the optimal set of millions of weights and biases that globally minimize a predefined loss function. This optimization is achieved through an iterative process involving forward propagation (calculating predictions), loss computation (measuring error), and backward propagation (calculating gradients via the chain rule). Optimization algorithms, such as Stochastic Gradient Descent (SGD) and Adam (Adaptive Moment Estimation), use these gradients to iteratively update the network parameters until convergence is achieved.", 0, 12),
    
    ("## 2.2 Standard Frameworks and Libraries", 2, 12),
    ("### 2.2.1 Python and PyTorch", 3, 12),
    ("Python serves as the primary programming language for this project due to its unparalleled ecosystem of machine learning libraries, rapid prototyping capabilities, and extensive community support. The core deep learning architecture is implemented utilizing PyTorch, a premier open-source machine learning framework developed primarily by Facebook's AI Research lab (FAIR).", 0, 12),
    ("PyTorch was selected over alternatives like TensorFlow due to its dynamic computational graph paradigm (often referred to as 'define-by-run'). This architecture allows the computational graph to be built dynamically on the fly during execution, offering exceptional flexibility for complex models and vastly simplifying the debugging process. PyTorch operates fundamentally on Tensors—multi-dimensional arrays similar to NumPy's ndarrays but optimized for massive parallel computation on GPUs. Furthermore, PyTorch's Autograd engine dynamically tracks operations performed on these tensors to automatically and efficiently compute complex gradients during backpropagation.", 0, 12),
    
    ("### 2.2.2 OpenCV (Open Source Computer Vision Library)", 3, 12),
    ("OpenCV is an extensive, open-source library specifically designed for high-performance computational efficiency and real-time computer vision applications. Within the DocForge data pipeline, OpenCV handles all fundamental low-level image processing operations required prior to tensor conversion.", 0, 12),
    ("Crucial operations include reading raw image binaries from disk using optimized decoders, performing necessary color space conversions (e.g., transforming images from the default BGR format to RGB), executing high-quality bicubic and bilinear image interpolations for resizing, and applying complex morphological operations (such as Canny edge detection) which are vital for processing and analyzing the ground truth forgery masks.", 0, 12),
    
    ("## 2.3 Attention Mechanisms (Generic)", 2, 12),
    ("A critical evolutionary step in modern deep learning architectures is the incorporation of Attention Mechanisms. Inspired directly by human cognitive visual attention—where we selectively focus on specific, salient regions of our visual field while actively suppressing irrelevant peripheral details—attention mechanisms enable neural networks to dynamically weigh the importance of different extracted features.", 0, 12),
    ("In computer vision, attention is generally categorized into two orthogonal dimensions: Spatial Attention and Channel Attention. Spatial Attention helps the network determine *where* to focus within the spatial dimensions of an image, generating a 2D weight map that highlights informative regions (such as forgery boundaries) and suppresses homogeneous background noise. Channel Attention determines *what* specific features to focus on by assigning dynamic weights to individual feature map channels, emphasizing the most discriminative learned filters.", 0, 12),
    ("By intelligently integrating these attention mechanisms, the DocForge model can selectively amplify subtle manipulation artifacts—such as microscopic noise inconsistencies, unnatural blending boundaries, and edge anomalies—thereby significantly enhancing both the sensitivity and precision of the overall forgery detection system.", 0, 12),
    
    ("# CHAPTER 3: SYSTEM ARCHITECTURE & IMPLEMENTATION", 1, 14),
    ("## 3.1 System Pipeline", 2, 12),
    ("The DocForge system is structured as a comprehensive, end-to-end processing pipeline that seamlessly transforms raw digital documents into highly localized forgery predictions. The pipeline is designed to be highly resilient against various forms of severe document degradation and standard benign image transformations.", 0, 12),
    ("The ingestion phase involves reading digital images (e.g., scanned documents, identity cards) in standard compression formats (JPEG, PNG). The data loading module precisely aligns the input image with its corresponding pixel-level ground truth mask for supervised training. ", 0, 12),
    ("Data Preprocessing and Augmentation is a crucial step to ensure the model's robustness and generalization capabilities against unseen data. All input images and masks are standardized to a fixed spatial resolution (e.g., 256x256 pixels) using carefully selected interpolation techniques to ensure uniform tensor dimensions for the neural network batches without destroying crucial forensic artifacts.", 0, 12),
    ("During training, extensive geometric transformations (random flips, rotations, scale-crops) and photometric distortions (random brightness, contrast adjustments, Gaussian blurring) are applied. This artificial expansion of the dataset simulates real-world document variations, scanning artifacts, and prevents the network from memorizing the training set.", 0, 12),
    
    ("## 3.2 Abstracted Architecture Description", 2, 12),
    ("The core intelligence of the DocForge system is encapsulated within a custom deep convolutional neural network architecture specifically tailored for manipulation detection. The architecture follows an advanced encoder-decoder structure heavily enhanced with specialized analysis modules.", 0, 12),
    ("The Feature Extraction Backbone acts as the primary sensory organ of the network. It comprises multiple dense layers of convolutions designed to extract hierarchical representations of the image, ranging from low-level edges and textures to high-level semantic structures.", 0, 12),
    ("The features extracted by the backbone are then routed into an Advanced Feature Fusion Pipeline. This proprietary module utilizes a Custom Dual-Branch Convolutional Network combined with a sophisticated Spatial-Channel Attention Mechanism. This attention mechanism is highly specialized; it dynamically emphasizes spatial discrepancies and structural anomalies that are strongly indicative of forgery, effectively filtering out authentic background patterns.", 0, 12),
    ("Finally, the fused, attention-weighted features are progressively upsampled through a decoding structure to restore the original spatial resolution. A prediction head applies a sigmoid activation function to yield the continuous, localized pixel-wise forgery probability map.", 0, 12),

    ("## 3.3 Implementation Snippets", 2, 12),
    ("Below is a representative snippet demonstrating the robust ingestion and preprocessing pipeline utilized within the project framework.", 0, 12),
    
    ("```python", -1, 10),
    ("import cv2\nimport numpy as np\nimport torch\nimport random\nfrom torch.utils.data import Dataset\nfrom PIL import Image, ImageEnhance\n\nclass ForgeryDataset(Dataset):\n    def __init__(self, samples, crop_size=256, is_train=True):\n        self.crop_size = crop_size\n        self.is_train = is_train\n        self.samples = samples\n        \n    def __len__(self):\n        return len(self.samples)\n\n    def __getitem__(self, idx):\n        img_path, mask_path = self.samples[idx]\n        img = cv2.imread(img_path)\n        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)\n        \n        if img.shape[:2] != mask.shape[:2]:\n            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)\n\n        mask = (mask > 127).astype(np.uint8)\n        edge = (cv2.Canny(mask * 255, 100, 200) > 0).astype(np.uint8)\n        \n        # Tensor Conversion abstracted...\n        return img_tensor, mask_tensor, edge_tensor", -1, 10),
    ("```", -1, 10),

    ("# CHAPTER 4: RESULTS AND DISCUSSIONS", 1, 14),
    ("## 4.1 Evaluation Metrics", 2, 12),
    ("To rigorously evaluate the performance of the DocForge manipulation detection system, a comprehensive suite of standard academic evaluation metrics was employed. These metrics provide a multifaceted view of the model's capabilities in correctly localizing forged pixels versus authentic background.", 0, 12),
    ("Precision (Positive Predictive Value) measures the accuracy of the positive predictions. It answers the question: Of all the pixels the model predicted as forged, what fraction was actually forged? High precision indicates a low rate of false alarms.", 0, 12),
    ("Recall (Sensitivity or True Positive Rate) measures the ability of the model to identify all actual positive cases. It answers the question: Of all the actually forged pixels, what fraction did the model successfully detect? In security applications like document forgery, high recall is often prioritized to ensure no manipulation slips through.", 0, 12),
    ("F1-Score is the harmonic mean of Precision and Recall. It provides a single metric that balances both concerns, which is particularly crucial in highly imbalanced datasets where the forged regions represent a very small percentage of the total document area.", 0, 12),
    ("Intersection over Union (IoU / Jaccard Index) is a stringent metric specifically used for segmentation tasks. It calculates the area of overlap between the predicted bounding box/mask and the ground truth mask, divided by the area of their union.", 0, 12),
    
    ("## 4.2 Visual Results Analysis", 2, 12),
    ("[INSERT LOSS/ACCURACY GRAPH HERE]", 0, 12),
    ("The training curves illustrate the steady minimization of the combined loss function across epochs, alongside corresponding improvements in validation F1-Score and IoU, indicating successful learning without severe overfitting.", 0, 12),
    ("[INSERT SIDE-BY-SIDE COMPARISON OF ORIGINAL VS FORGED MASK HERE]", 0, 12),
    ("The visual comparison explicitly demonstrates the model's efficacy. The generated predicted mask tightly correlates with the ground truth mask, successfully isolating the manipulated elements while ignoring the complex, authentic background textures of the document.", 0, 12),
    
    ("# CHAPTER 5: CONCLUSIONS AND FUTURE SCOPE", 1, 14),
    ("## 5.1 Conclusion", 2, 12),
    ("The DocForge project successfully designed, implemented, and evaluated an advanced deep-learning pipeline for the detection and localization of digital document manipulations. By engineering a custom deep learning architecture featuring a robust feature extraction backbone and a specialized attention-based fusion module, the system demonstrated significant efficacy in identifying subtle forgery artifacts.", 0, 12),
    ("The integration of a rigorous data augmentation pipeline, combined with a specialized dual-loss optimization strategy, enabled the model to learn complex representations and achieve high accuracy on evaluation datasets. The resulting pixel-wise localization maps provide clear, interpretable evidence of manipulation.", 0, 12),
