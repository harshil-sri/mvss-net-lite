import os
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_overlay(original_path: str, mask_path: str, output_path: str, regions: list = None):
    orig = Image.open(original_path).convert("RGBA")

    if os.path.exists(mask_path):
        mask = Image.open(mask_path).convert("L").resize(orig.size)
    else:
        mask = Image.new("L", orig.size, 0)
        if regions:
            draw = ImageDraw.Draw(mask)
            for r in regions:
                bbox = r.get("bbox", {})
                x, y, w, h = bbox.get("x", 0), bbox.get("y", 0), bbox.get("w", 0), bbox.get("h", 0)
                if w > 0 and h > 0:
                    draw.rectangle([x, y, x + w, y + h], fill=255)

    alpha = mask.point(lambda p: int(p * 0.45))
    red_layer = Image.new("RGBA", orig.size, (255, 0, 0, 0))
    red_layer.putalpha(alpha)

    overlay = Image.alpha_composite(orig, red_layer)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    overlay.save(output_path)


def generate_chart(regions: list, output_path: str):
    ids = [r["region_id"] for r in regions] or ["-"]
    conf = [r.get("local_confidence", 0) for r in regions] or [0]
    edge = [r.get("edge_consistency_score", 0) for r in regions] or [0]
    x = range(len(ids))

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar([i - 0.2 for i in x], conf, 0.4, label="Confidence", color="#ff6b6b")
    ax.bar([i + 0.2 for i in x], edge, 0.4, label="Edge Consistency", color="#4ecdc4")
    ax.set_xticks(list(x))
    ax.set_xticklabels(ids)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
