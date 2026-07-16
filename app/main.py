import logging

from fastapi import FastAPI

from app.api.websocket import router as websocket_router

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="2care Voice Orchestrator")

app.include_router(websocket_router)


@app.get("/")
async def health_check():
    return {"status": "ok"}
