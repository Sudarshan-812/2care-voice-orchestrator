import logging
import time
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.cliniko import ClinikoAPIError, cliniko_client
from app.services.db import db
from app.services.llm import llm_client

logger = logging.getLogger(__name__)

router = APIRouter()

RECOVERY_WINDOW = timedelta(minutes=15)


@router.websocket("/llm-websocket/{call_id}")
async def llm_websocket(websocket: WebSocket, call_id: str) -> None:
    await websocket.accept()
    logger.info("WebSocket connection accepted for call_id=%s", call_id)

    call_context: str | None = None
    from_number: str | None = None

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
                    previous_state = await db.get_call_state(from_number)
                    if previous_state and previous_state["status"] == "active":
                        elapsed = datetime.now(timezone.utc) - previous_state["last_updated"]
                        if elapsed < RECOVERY_WINDOW:
                            context_parts.append(
                                "The previous call dropped unexpectedly. You already have "
                                f"context: {previous_state['context_snapshot']}. Apologize for "
                                "the disconnect and resume where you left off."
                            )
                            logger.warning(
                                "Recovered dropped call for call_id=%s from=%s previous_call_id=%s elapsed=%s",
                                call_id,
                                from_number,
                                previous_state["call_id"],
                                elapsed,
                            )

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

                try:
                    businesses = await cliniko_client.get_businesses()
                except ClinikoAPIError as exc:
                    logger.error("Failed to fetch businesses for call_id=%s: %s", call_id, exc)
                    businesses = []

                if businesses:
                    branches = ", ".join(f"ID {b['id']} ({b['name']})" for b in businesses)
                    context_parts.append(
                        f"The clinic has the following locations (branches): {branches}."
                    )
                    logger.info("Loaded %d branches for call_id=%s", len(businesses), call_id)

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

            if from_number:
                await db.upsert_call_state(call_id, from_number, "active", call_context)

            elapsed_ms = (time.monotonic() - start) * 1000
            logger.info(
                "Processed interaction_type=%s call_id=%s in %.1fms",
                interaction_type,
                call_id,
                elapsed_ms,
            )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for call_id=%s", call_id)
        if from_number:
            await db.upsert_call_state(call_id, from_number, "completed", call_context)
