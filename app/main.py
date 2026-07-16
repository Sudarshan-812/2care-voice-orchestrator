import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.websocket import router as websocket_router
from app.services.db import db

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db()
    app.state.db_pool = db.pool
    try:
        yield
    finally:
        await db.close()


app = FastAPI(title="2care Voice Orchestrator", lifespan=lifespan)

app.include_router(websocket_router)


@app.get("/")
async def health_check():
    return {"status": "ok"}
