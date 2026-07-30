import os
import time
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import numpy as np
import cv2
import sys

# Ensure project root is in sys.path for importing model definitions
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.network import MVSSNetLite

# Global cached model instance
_MODEL_INSTANCE = None
_MODEL_PATH_LOADED = None

def get_model(model_path: str = None):
    global _MODEL_INSTANCE, _MODEL_PATH_LOADED
    if model_path is None:
        model_path = os.path.join(PROJECT_ROOT, "backend", "stage2_mvss_lite_ep5.pt")
        if not os.path.exists(model_path):
            model_path = os.path.join(PROJECT_ROOT, "stage2_mvss_lite_ep5.pt")

    if _MODEL_INSTANCE is not None and _MODEL_PATH_LOADED == model_path:
        return _MODEL_INSTANCE

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    print(f"Loading MVSSNetLite checkpoint from: {model_path}")
    model = MVSSNetLite()
    checkpoint = torch.load(model_path, map_location="cpu")
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    _MODEL_INSTANCE = model
    _MODEL_PATH_LOADED = model_path
    return model


def preprocess_image(image_path: str, target_size=(512, 512)):
    """
    Loads an image, resizes it to target_size, converts to Tensor and normalizes.
    Returns:
        input_tensor: PyTorch tensor (1, 3, H, W)
        orig_size: (width, height)
    """
    orig_img = Image.open(image_path).convert("RGB")
    orig_size = orig_img.size  # (W, H)

    transform = T.Compose([
        T.Resize(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(orig_img).unsqueeze(0)  # (1, 3, H, W)
    return input_tensor, orig_size


def predict_document(image_path: str, prediction_id: str, filename: str, threshold: float = 0.45) -> dict:
    """
    Runs MVSSNetLite model inference on the uploaded image.
    Generates prediction mask PNG and computes region bounding boxes and scores.
    """
    start_time = time.time()
    
    model = get_model()
    input_tensor, (orig_w, orig_h) = preprocess_image(image_path, target_size=(512, 512))

    with torch.no_grad():
        seg_logits, edge_logits = model(input_tensor)
        seg_prob = torch.sigmoid(seg_logits).squeeze().cpu().numpy()  # 2D array (512, 512)
        edge_prob = torch.sigmoid(edge_logits).squeeze().cpu().numpy() # 2D array (512, 512)

    # Resize probabilities back to original dimensions
    seg_prob_orig = cv2.resize(seg_prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    edge_prob_orig = cv2.resize(edge_prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # Create binary mask (0 or 255)
    binary_mask = (seg_prob_orig >= threshold).astype(np.uint8) * 255

    # Save mask artifact
    mask_dir = "app/static/generated"
    os.makedirs(mask_dir, exist_ok=True)
    mask_filename = f"{prediction_id}_mask.png"
    mask_path = os.path.join(mask_dir, mask_filename)
    cv2.imwrite(mask_path, binary_mask)

    # Find contours for manipulated regions
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    manipulated_regions = []
    region_idx = 1
    
    # Filter small noisy contours (min area threshold: 0.05% of image size)
    min_area = (orig_w * orig_h) * 0.0005

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        
        # Crop region probability to calculate local confidence
        region_prob = seg_prob_orig[y:y+h, x:x+w]
        local_conf = float(np.mean(region_prob)) if region_prob.size > 0 else 0.5
        
        # Crop edge probability for edge consistency score
        region_edge = edge_prob_orig[y:y+h, x:x+w]
        edge_score = float(np.mean(region_edge)) if region_edge.size > 0 else 0.5

        manipulated_regions.append({
            "region_id": f"r{region_idx}",
            "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "local_confidence": round(local_conf, 2),
            "edge_consistency_score": round(edge_score, 2)
        })
        region_idx += 1

    # Determine global verdict and overall confidence
    max_prob = float(np.max(seg_prob_orig))
    verdict = "Forged" if len(manipulated_regions) > 0 or max_prob >= threshold else "Authentic"
    overall_confidence = round(max_prob, 2) if verdict == "Forged" else round(1.0 - max_prob, 2)

    inference_ms = int((time.time() - start_time) * 1000)

    prediction = {
        "prediction_id": prediction_id,
        "filename": filename,
        "verdict": verdict,
        "confidence": overall_confidence,
        "manipulated_regions": manipulated_regions,
        "artifacts": {
            "mask_path": f"app/static/generated/{prediction_id}_mask.png"
        },
        "model_meta": {
            "model_version": "MVSS-Net-Lite (Stage2 Ep5)",
            "inference_time_ms": inference_ms,
            "detection_mode": "model"
        },
        "_upload_path": image_path
    }

    return prediction
