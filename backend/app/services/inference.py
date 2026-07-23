import os
import cv2
import glob
import torch
import numpy as np
from torchvision import transforms

# Add root directory to path so we can import model
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from model.network import MVSSNetLite

# Global model instance
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None

def load_model():
    global model
    if model is not None:
        return model
        
    model = MVSSNetLite().to(device)
    
    # Robust path resolution for checkpoints
    base_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(base_dir, "../../../"))
    checkpoint_dir = os.path.join(project_root, "model", "checkpoints")
    
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
    if checkpoints:
        # Sort by modification time to get the latest
        latest_ckpt = max(checkpoints, key=os.path.getmtime)
        print(f"Loading checkpoint: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("WARNING: No checkpoints found. Using untrained model weights.")
        
    model.eval()
    return model

def analyze_image(img_path: str, prediction_id: str):
    """
    Runs MVSSNetLite on the image and extracts forging metrics.
    """
    model = load_model()
    
    # Load and preprocess image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
        
    original_h, original_w = img.shape[:2]
    
    # Same transform as training
    img_resized = cv2.resize(img, (256, 256))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tensor_img = transform(img_rgb).unsqueeze(0).to(device)
    
    # Inference
    t0 = cv2.getTickCount()
    with torch.no_grad():
        pred_seg, pred_edge = model(tensor_img)
        
        # Apply sigmoid to get probabilities
        prob_seg = torch.sigmoid(pred_seg).squeeze().cpu().numpy()
        prob_edge = torch.sigmoid(pred_edge).squeeze().cpu().numpy()
        
    t1 = cv2.getTickCount()
    inference_time_ms = int((t1 - t0) * 1000 / cv2.getTickFrequency())
    
    # Thresholding
    mask_bin = (prob_seg > 0.5).astype(np.uint8) * 255
    edge_bin = (prob_edge > 0.5).astype(np.uint8) * 255
    
    # Resize mask back to original dimensions for saving and bbox calculations
    mask_original_size = cv2.resize(mask_bin, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    edge_original_size = cv2.resize(edge_bin, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    
    # Save artifacts
    mask_filename = f"{prediction_id}_mask.png"
    edge_filename = f"{prediction_id}_edge.png"
    
    # Robust path resolution
    base_dir = os.path.dirname(__file__)
    generated_dir = os.path.abspath(os.path.join(base_dir, "../../app/static/generated"))
    os.makedirs(generated_dir, exist_ok=True)
    
    mask_save_path = os.path.join(generated_dir, mask_filename)
    edge_save_path = os.path.join(generated_dir, edge_filename)
    
    cv2.imwrite(mask_save_path, mask_original_size)
    cv2.imwrite(edge_save_path, edge_original_size)
    
    # Find manipulated regions using contours
    contours, _ = cv2.findContours(mask_original_size, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    regions = []
    global_forgery_confidence = 0.0
    
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > 100:  # Filter out tiny noise dots
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Calculate local confidence based on probability map in this region
            # Map original bbox to 256x256 to sample the probability map
            x_256 = int(x * 256 / original_w)
            y_256 = int(y * 256 / original_h)
            w_256 = max(1, int(w * 256 / original_w))
            h_256 = max(1, int(h * 256 / original_h))
            
            region_prob = prob_seg[y_256:y_256+h_256, x_256:x_256+w_256]
            local_conf = float(np.mean(region_prob)) if region_prob.size > 0 else 0.5
            
            global_forgery_confidence = max(global_forgery_confidence, local_conf)
            
            regions.append({
                "region_id": f"r{i+1}",
                "bbox": {"x": x, "y": y, "w": w, "h": h},
                "local_confidence": round(local_conf, 3),
                "edge_consistency_score": round(float(np.mean(prob_edge)), 3) # Simplify edge score
            })
            
    verdict = "Forged" if len(regions) > 0 and global_forgery_confidence > 0.6 else "Authentic"
    if verdict == "Authentic":
        global_forgery_confidence = 1.0 - float(np.mean(prob_seg)) # Confidence it is real
        
    return {
        "verdict": verdict,
        "confidence": round(global_forgery_confidence, 3),
        "manipulated_regions": regions,
        "artifacts": {
            "mask_path": f"app/static/generated/{mask_filename}",
            "edge_path": f"app/static/generated/{edge_filename}"
        },
        "model_meta": {
            "model_version": "mvss-lite-v1",
            "inference_time_ms": inference_time_ms
        }
    }
