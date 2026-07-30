import os
import time
import uuid
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import RedirectResponse
from app.services.store import save_prediction
from app.services.inference_service import predict_document

router = APIRouter()
UPLOAD_DIR = "app/static/uploads"

@router.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    upload_path = os.path.join(UPLOAD_DIR, filename)

    with open(upload_path, "wb") as f:
        f.write(await file.read())

    prediction_id = f"pred_{int(time.time())}"
    
    print(f"Running MVSSNetLite PyTorch model inference for: {filename}")
    prediction = predict_document(upload_path, prediction_id, file.filename)

    save_prediction(prediction_id, prediction)
    return RedirectResponse(url=f"/report/{prediction_id}", status_code=303)

