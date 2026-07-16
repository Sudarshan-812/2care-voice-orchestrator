import logging
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.cliniko import ClinikoAPIError, cliniko_client
from app.services.llm import llm_client

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/llm-websocket/{call_id}")
async def llm_websocket(websocket: WebSocket, call_id: str) -> None:
    await websocket.accept()
    logger.info("WebSocket connection accepted for call_id=%s", call_id)

    call_context: str | None = None

    try:
        while True:
            message = await websocket.receive_json()
            start = time.monotonic()
            interaction_type = message.get("interaction_type")

            if interaction_type == "call_details":
                call_data = message.get("call", {})
                from_number = call_data.get("from_number")
                logger.info(
                    "Call started: call_id=%s from=%s to=%s",
                    call_id,
                    from_number,
                    call_data.get("to_number"),
                )

                context_parts: list[str] = []

                if from_number:
                    try:
                        existing_patients = await cliniko_client.get_patients_by_phone(from_number)
                    except ClinikoAPIError as exc:
                        logger.error(
                            "Patient lookup failed for call_id=%s from=%s: %s", call_id, from_number, exc
                        )
                        existing_patients = []

                    if existing_patients:
                        names = ", ".join(
                            f"{p.get('first_name', '')} {p.get('last_name', '')}".strip()
                            for p in existing_patients
                        )
                        context_parts.append(
                            f"This phone number is associated with returning patients: {names}. "
                            "Greet them, but DO NOT assume which one is speaking. Ask for their "
                            "name to confirm."
                        )
                        logger.info(
                            "Returning patient(s) detected for call_id=%s: %s", call_id, names
                        )

                try:
                    appointment_types = await cliniko_client.get_appointment_types()
                except ClinikoAPIError as exc:
                    logger.error(
                        "Failed to fetch appointment types for call_id=%s: %s", call_id, exc
                    )
                    appointment_types = []

                if appointment_types:
                    services = ", ".join(
                        f"ID {t['id']} ({t['name']}, {t['duration_in_minutes']} mins)"
                        for t in appointment_types
                    )
                    context_parts.append(f"The clinic offers the following services: {services}.")
                    logger.info(
                        "Loaded %d appointment types for call_id=%s", len(appointment_types), call_id
                    )

                call_context = " ".join(context_parts) or None

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
                await llm_client.draft_response(message, websocket, clinic_context=call_context)

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
