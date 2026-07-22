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
    
    # Run the live model inference
    try:
        from app.services.inference import analyze_image
        prediction = analyze_image(upload_path, prediction_id)
    except Exception as e:
        print(f"Inference error: {e}")
        # Fallback in case of failure
        prediction = {
            "verdict": "Error",
            "confidence": 0.0,
            "manipulated_regions": [],
            "artifacts": {},
            "model_meta": {"error": str(e)}
        }
        
    # Append dynamic upload info
    prediction["prediction_id"] = prediction_id
    prediction["filename"] = filename
    prediction["_upload_path"] = upload_path

    save_prediction(prediction_id, prediction)
    return RedirectResponse(url=f"/report/{prediction_id}", status_code=303)

