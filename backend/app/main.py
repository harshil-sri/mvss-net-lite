import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.routes import analyze, report, chat

os.makedirs("app/static/uploads", exist_ok=True)
os.makedirs("app/static/generated", exist_ok=True)

app = FastAPI(title="Document Forgery Analysis")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

app.include_router(analyze.router)
app.include_router(report.router)
app.include_router(chat.router)


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})
