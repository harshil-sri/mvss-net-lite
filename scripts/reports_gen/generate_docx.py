import os
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
import re

doc = Document()

# Set default font to Times New Roman
style = doc.styles['Normal']
font = style.font
font.name = 'Times New Roman'
font.size = Pt(12)

def add_paragraph(text, size=12, bold=False, align=WD_ALIGN_PARAGRAPH.LEFT, spacing=1.0):
    p = doc.add_paragraph()
    p.alignment = align
    if spacing == 1.5:
        p.paragraph_format.line_spacing = 1.5
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(size)
    run.font.bold = bold
    return p

# Front matter
add_paragraph("COVER PAGE", size=12, align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()
add_paragraph("DOCFORGE", size=24, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()
add_paragraph("Submitted in partial fulfillment of the Requirements for the award of Degree of Bachelor of Technology in Artificial Intelligence and Machine Learning (AI/ML)", size=16, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()
doc.add_paragraph()
add_paragraph("Submitted By", size=14, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
add_paragraph("Name: Harshil Srivastav\nUniversity Roll No.: 35817711624\nSemester: 5th\nAcademic Year: 2026-27", size=14, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)

doc.add_page_break()

add_paragraph("DECLARATION", size=16, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()
add_paragraph('I hereby declare that the Internship Report entitled "DocForge" is an authentic record of my own work as requirements of Internship during the period from 15.06.2026 to 31.07.2026 for the award of degree of B.Tech. (AIML) School of Engineering and Technology, VIPS-TC.', size=12, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
doc.add_paragraph()
doc.add_paragraph()
add_paragraph("(Signature of student)\nHarshil Srivastav\n35817711624\nDate: 25 September 2026", size=12)

doc.add_page_break()

add_paragraph("CERTIFICATE", size=16, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()
add_paragraph("[INSERT C-DAC CERTIFICATE IMAGE (image_4bdd7b.jpg) HERE]", size=12, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()
add_paragraph("This is to certify that Mr. Harshil Srivastav has completed Internship during the period from 15.06.2026 to 31.07.2026 in our Organization / Industry as a Partial Fulfillment of Degree of Bachelor of Technology in AIML. He was trained in the field of Artificial Intelligence and Machine Learning / Document Forgery Detection.", size=12, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
doc.add_paragraph()
doc.add_paragraph()
add_paragraph("Signature & Seal of Training Manager\nMr. Tushar Kanti Patnaik, Scientist F", size=12)

doc.add_page_break()

add_paragraph("ACKNOWLEDGEMENT", size=14, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()
add_paragraph("I would like to express my profound gratitude to the Centre for Development of Advanced Computing (C-DAC), Noida, for providing me with the invaluable opportunity to pursue my summer internship. This experience has significantly enriched my practical understanding of machine learning and computer vision.", size=12, align=WD_ALIGN_PARAGRAPH.JUSTIFY, spacing=1.5)
doc.add_paragraph()
add_paragraph("I am deeply indebted to my project guide, Mr. Tushar Kanti Patnaik, Scientist F, for his continuous guidance, encouragement, and technical mentorship throughout the development of the \"DocForge\" project. His insights were instrumental in navigating the complexities of deep learning pipelines.", size=12, align=WD_ALIGN_PARAGRAPH.JUSTIFY, spacing=1.5)
doc.add_paragraph()
add_paragraph("I also extend my sincere thanks to the faculty and administration of Vivekananda Institute of Professional Studies - Technical Campus (VIPS-TC), Pitampura, for equipping me with the foundational knowledge required to undertake this training and for their continuous academic support. Finally, I would like to thank my family and peers for their unwavering encouragement throughout this endeavor.", size=12, align=WD_ALIGN_PARAGRAPH.JUSTIFY, spacing=1.5)

doc.add_page_break()

add_paragraph("ABOUT COMPANY", size=14, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()
add_paragraph("The Centre for Development of Advanced Computing (C-DAC) is the premier research and development organization of the Ministry of Electronics and Information Technology (MeitY), Government of India. Established to advance the country's IT and electronics sector, C-DAC engages in cutting-edge research in areas such as High-Performance Computing (HPC), Artificial Intelligence, Cyber Security, and Software Technologies. The Noida facility, located in the Institutional Area of Sector 62, actively drives technological innovation, education, and specialized training programs designed to bridge the gap between academic learning and industry demands.", size=12, align=WD_ALIGN_PARAGRAPH.JUSTIFY, spacing=1.5)

doc.add_page_break()

# Read the rest of the body from the generated markdown artifact
md_path = "/home/harshil/.gemini/antigravity-ide/brain/4d734d0a-9ed6-4d39-bb88-9c03960e4cee/docforge_internship_report.md"

with open(md_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

in_body = False
in_code = False
code_lines = []

for line in lines:
    if line.strip() == "# CHAPTER 1: INTRODUCTION TO PROJECT":
        in_body = True
    
    if not in_body:
        continue
        
    stripped = line.strip()
    
    if stripped.startswith("```"):
        if in_code:
            in_code = False
            # write code block
            p = doc.add_paragraph()
            run = p.add_run("".join(code_lines))
            run.font.name = 'Courier New'
            run.font.size = Pt(10)
            code_lines = []
        else:
            in_code = True
        continue
        
    if in_code:
        code_lines.append(line)
        continue
        
    if not stripped:
        doc.add_paragraph()
        continue
        
    if stripped.startswith("# "):
        doc.add_page_break()
        add_paragraph(stripped[2:], size=16, bold=True)
    elif stripped.startswith("## "):
        add_paragraph(stripped[3:], size=14, bold=True)
    elif stripped.startswith("### "):
        add_paragraph(stripped[4:], size=12, bold=True)
    else:
        # Handling basic bold inline formatting
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        
        parts = re.split(r'(\*\*.*?\*\*)', stripped)
        for part in parts:
            if part.startswith('**') and part.endswith('**'):
                run = p.add_run(part[2:-2])
                run.bold = True
            else:
                run = p.add_run(part)

downloads_path = "/mnt/c/Users/Harshil/Downloads/DocForge_Internship_Report.docx"
doc.save(downloads_path)
print(f"Saved to {downloads_path}")
