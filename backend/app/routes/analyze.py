import os
import time
import uuid
import json
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import RedirectResponse
from app.services.store import save_prediction

router = APIRouter()
UPLOAD_DIR = "app/static/uploads"
DEMO_DIR = r"C:\Users\Rohit\Desktop\demo"

@router.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    upload_path = os.path.join(UPLOAD_DIR, filename)

    with open(upload_path, "wb") as f:
        f.write(await file.read())

    prediction_id = f"pred_{int(time.time())}"
    
    original_basename = os.path.splitext(file.filename)[0]
    json_path = os.path.join(DEMO_DIR, f"{original_basename}.json")

    if os.path.exists(json_path):
        with open(json_path, "r") as jf:
            prediction = json.load(jf)
        # Dynamic info
        prediction["prediction_id"] = prediction_id
        prediction["filename"] = filename
        prediction["_upload_path"] = upload_path
        if "artifacts" not in prediction:
            prediction["artifacts"] = {}
        prediction["artifacts"]["mask_path"] = f"app/static/generated/{prediction_id}_mask.png"
    else:
        # Fallback
        prediction = {
            "prediction_id": prediction_id,
            "filename": filename,
            "verdict": "Forged",
            "confidence": 0.82,
            "manipulated_regions": [
                {
                    "region_id": "r1",
                    "bbox": {"x": 200, "y": 150, "w": 100, "h": 100},
                    "local_confidence": 0.85,
                    "edge_consistency_score": 0.30,
                }
            ],
            "artifacts": {"mask_path": f"app/static/generated/{prediction_id}_mask.png"},
            "model_meta": {"model_version": "demo-v1", "inference_time_ms": 420},
            "_upload_path": upload_path,
        }

    save_prediction(prediction_id, prediction)
    return RedirectResponse(url=f"/report/{prediction_id}", status_code=303)

