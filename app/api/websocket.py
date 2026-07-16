import logging
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/llm-websocket/{call_id}")
async def llm_websocket(websocket: WebSocket, call_id: str) -> None:
    await websocket.accept()
    logger.info("WebSocket connection accepted for call_id=%s", call_id)

    try:
        while True:
            message = await websocket.receive_json()
            start = time.monotonic()
            interaction_type = message.get("interaction_type")

            if interaction_type == "call_details":
                call_data = message.get("call", {})
                logger.info(
                    "Call started: call_id=%s from=%s to=%s",
                    call_id,
                    call_data.get("from_number"),
                    call_data.get("to_number"),
                )

            elif interaction_type == "ping":
                await websocket.send_json(
                    {"interaction_type": "ping_pong", "timestamp": message["timestamp"]}
                )

            elif interaction_type == "update":
                logger.info(
                    "Transcript update: call_id=%s transcript=%s",
                    call_id,
                    message.get("transcript"),
                )

            elif interaction_type == "response_required":
                logger.info(
                    "Response required: call_id=%s response_id=%s",
                    call_id,
                    message.get("response_id"),
                )
                await websocket.send_json(
                    {
                        "response_id": message["response_id"],
                        "content": "Hello, I am the clinic assistant. I am currently connected.",
                        "content_complete": True,
                    }
                )

            elif interaction_type == "reminder_required":
                logger.info(
                    "Reminder required: call_id=%s response_id=%s",
                    call_id,
                    message.get("response_id"),
                )

            else:
                logger.warning(
                    "Unhandled interaction_type=%s call_id=%s message=%s",
                    interaction_type,
                    call_id,
                    message,
                )

            elapsed_ms = (time.monotonic() - start) * 1000
            logger.info(
                "Processed interaction_type=%s call_id=%s in %.1fms",
                interaction_type,
                call_id,
                elapsed_ms,
            )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for call_id=%s", call_id)
