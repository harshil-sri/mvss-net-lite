import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.store import get_prediction
from app.services.llm_client import call_llm

router = APIRouter()


class ChatRequest(BaseModel):
    question: str
    history: list[dict] = []


@router.post("/chat/{prediction_id}")
async def chat_with_prediction(prediction_id: str, req: ChatRequest):
    prediction = get_prediction(prediction_id)
    if not prediction:
        raise HTTPException(404, "Prediction not found")

    system = (
        "Answer ONLY using the JSON below. If the answer isn't in it, say you don't know.\n"
        f"{json.dumps(prediction)}"
    )
    convo = "\n".join(f'{m["role"]}: {m["content"]}' for m in req.history)
    prompt = f"{convo}\nuser: {req.question}" if convo else req.question

    return {"answer": call_llm(prompt, system=system)}
