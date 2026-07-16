from fastapi import FastAPI, WebSocket

app = FastAPI(title="2care Voice Orchestrator")


@app.get("/")
async def health_check():
    return {"status": "ok"}


@app.websocket("/llm-websocket")
async def llm_websocket(websocket: WebSocket):
    await websocket.accept()
