from fastapi import APIRouter, Request, HTTPException
from fastapi.templating import Jinja2Templates
from app.services.store import get_prediction
from app.services.report_builder import generate_overlay, generate_chart
from app.services.llm_client import call_llm

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/report/{prediction_id}")
async def view_report(request: Request, prediction_id: str):
    prediction = get_prediction(prediction_id)
    if not prediction:
        raise HTTPException(404, "Prediction not found")

    overlay_path = f"app/static/generated/{prediction_id}_overlay.png"
    chart_path = f"app/static/generated/{prediction_id}_chart.png"

    generate_overlay(prediction["_upload_path"], prediction["artifacts"]["mask_path"], overlay_path, regions=prediction.get("manipulated_regions", []))
    generate_chart(prediction["manipulated_regions"], chart_path)

    def url(path: str) -> str:
        return "/" + path.replace("app/", "")

    return templates.TemplateResponse(request=request, name="report.html", context={
        "prediction": prediction,
        "original_image_url": url(prediction["_upload_path"]),
        "mask_image_url": url(prediction["artifacts"]["mask_path"]),
        "overlay_image_url": url(overlay_path),
        "chart_image_url": url(chart_path),
        "ai_summary": "Generating AI summary...",
    })

@router.get("/summary/{prediction_id}")
async def get_summary(prediction_id: str):
    prediction = get_prediction(prediction_id)
    if not prediction:
        raise HTTPException(404, "Prediction not found")
        
    prompt_text = f"""Please analyze this document forensics report data and write a short, professional, natural-language summary. Do NOT output any JSON.
Filename: {prediction.get('filename')}
Verdict: {prediction.get('verdict')}
Confidence: {prediction.get('confidence')}
Number of Manipulated Regions: {len(prediction.get('manipulated_regions', []))}
"""
    summary = call_llm(
        prompt_text,
        system="You are an expert document forensics assistant. Provide a concise, factual summary in plain text. Never output JSON.",
    )
    return {"summary": summary}
