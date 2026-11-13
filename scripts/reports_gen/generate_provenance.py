import json
import hashlib
import os

def get_hash(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

provenance = {
    "inputs": {
        "idea.md": get_hash("workspace/inputs/idea.md"),
        "experimental_log.md": get_hash("workspace/inputs/experimental_log.md"),
        "template.tex": get_hash("workspace/inputs/template.tex"),
        "conference_guidelines.md": get_hash("workspace/inputs/conference_guidelines.md")
    },
    "outline": get_hash("workspace/outline.json"),
    "refs": get_hash("workspace/refs.bib"),
    "figures": {
        fig: get_hash(f"workspace/figures/{fig}") for fig in os.listdir("workspace/figures") if fig.endswith('.png')
    } if os.path.exists("workspace/figures") else {},
    "final": {
        "paper.tex": get_hash("workspace/final/paper.tex"),
        "paper.pdf": get_hash("workspace/final/paper.pdf")
    }
}

with open("workspace/provenance.json", "w") as f:
    json.dump(provenance, f, indent=2)
