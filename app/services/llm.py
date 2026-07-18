import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any

from openai import AsyncOpenAI

from app.core.config import settings
from app.services.cliniko import ClinikoAPIError, cliniko_client
from app.services.db import db

logger = logging.getLogger(__name__)

aclient = AsyncOpenAI(api_key=settings.GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

SYSTEM_PROMPT = """You are a receptionist for a multi-branch clinic.

- You speak naturally in English and Hindi. If the caller code-switches (mixes English and Hindi in the same conversation or sentence), you must reply in that exact same conversational mix — match their language blend, do not force pure English or pure Hindi.
- NEVER guess or assume appointment availability. ALWAYS call the `check_availability` tool whenever the caller asks about or proposes a time.
- Always read the current date from the "Today is..." line in your context and use that exact year when computing start_date/end_date (and any other date) for `check_availability`, `book_appointment`, and `reschedule_appointment`. Never reuse a year from an example in a tool description, and never assume a year from your own training data.
- `check_availability`, `book_appointment`, and `reschedule_appointment` all require an exact business_id, practitioner_id, and appointment_type_id. NEVER guess or default any of these — there is no "default branch" or "default doctor". If the caller hasn't told you which branch, which practitioner, or which service they want, you MUST ask them before calling any of these tools. Use the branch list, practitioner list, and service list in your context to offer real choices by name.
- ALWAYS ask for the caller's full name before booking an appointment.
- If the caller changes their requested time to something different from what you last checked, you MUST call `check_availability` again for the new time. Do not call `check_availability` again just to re-confirm a slot you already offered and the caller already accepted — re-stating or confirming the same time is not a change.
- You must only book appointments for the services listed in your context. Use the exact appointment_type_id provided.
- You manage bookings across multiple branches. The available branches are listed in your context. If the caller doesn't care which branch, call `check_availability` once per branch listed in your context (each call still needs a real practitioner_id and appointment_type_id) and tell them which branch has the earliest opening. There is no way to search "all branches" in a single call.
- If the caller wants to cancel or reschedule an existing appointment and you do not already know its appointment_id, call `get_patient_appointments` first to look up their bookings and confirm which one they mean before acting.
- To reschedule an appointment, call the single `reschedule_appointment` tool with the existing appointment_id plus the new slot's details. Do NOT call `cancel_appointment` and `book_appointment` separately for a reschedule — `reschedule_appointment` handles the safe ordering (book the new slot, then cancel the old one) itself.
- Once you have offered a slot and the user has provided their name and phone number to book it, you MUST immediately call the `book_appointment` tool. DO NOT call `check_availability` again.
"""

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": (
                "Check REAL practitioner availability (their actual working hours and "
                "existing bookings) for a date range and time preference, at one specific "
                "branch and with one specific practitioner. business_id, practitioner_id, "
                "and appointment_type_id are all required — never guess them; ask the "
                "caller if any are unknown. To check multiple branches, call this once per "
                "branch."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "business_id": {
                        "type": "integer",
                        "description": (
                            "The Cliniko business ID for the branch to check, taken exactly "
                            "from the branch list in your context."
                        ),
                    },
                    "practitioner_id": {
                        "type": "integer",
                        "description": (
                            "The Cliniko practitioner ID for the doctor to check, taken "
                            "exactly from the practitioner list in your context."
                        ),
                    },
                    "appointment_type_id": {
                        "type": "integer",
                        "description": (
                            "The Cliniko appointment type ID for the requested service, "
                            "taken exactly from the service list in your context."
                        ),
                    },
                    "start_date": {
                        "type": "string",
                        "description": (
                            "ISO 8601 start of the search window. Use the current year from "
                            "the 'Today is...' date in your context, never a year from an "
                            "example, e.g. 2026-06-01T00:00:00Z."
                        ),
                    },
                    "end_date": {
                        "type": "string",
                        "description": (
                            "ISO 8601 end of the search window. Use the current year from "
                            "the 'Today is...' date in your context, never a year from an "
                            "example, e.g. 2026-06-02T00:00:00Z."
                        ),
                    },
                    "time_preference": {
                        "type": "string",
                        "description": "The caller's preferred time of day, e.g. 'morning', 'afternoon', '3pm'.",
                    },
                },
                "required": [
                    "business_id",
                    "practitioner_id",
                    "appointment_type_id",
                    "start_date",
                    "end_date",
                    "time_preference",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": (
                "Book an appointment with a practitioner. Only call this after confirming "
                "availability with check_availability and after obtaining the caller's full name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "first_name": {"type": "string", "description": "The caller's first name."},
                    "last_name": {"type": "string", "description": "The caller's last name."},
                    "phone_number": {"type": "string", "description": "The caller's phone number."},
                    "practitioner_id": {
                        "type": "integer",
                        "description": (
                            "The Cliniko practitioner ID. Must be the exact practitioner_id "
                            "you just confirmed availability for — never guess."
                        ),
                    },
                    "appointment_type_id": {
                        "type": "integer",
                        "description": (
                            "The Cliniko appointment type ID for the requested service, taken "
                            "exactly from the list of services provided in your context."
                        ),
                    },
                    "business_id": {
                        "type": "integer",
                        "description": (
                            "The Cliniko business ID for the branch. Must be the exact "
                            "business_id you just confirmed availability for — never guess."
                        ),
                    },
                    "start_time": {"type": "string", "description": "ISO 8601 appointment start time."},
                    "end_time": {"type": "string", "description": "ISO 8601 appointment end time."},
                },
                "required": [
                    "first_name",
                    "last_name",
                    "phone_number",
                    "practitioner_id",
                    "appointment_type_id",
                    "business_id",
                    "start_time",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reschedule_appointment",
            "description": (
                "Atomically move an existing appointment to a new slot: books the new slot "
                "first, and only cancels the old appointment if the new booking succeeds, "
                "so the caller is never left with nothing booked. Use this instead of "
                "calling cancel_appointment and book_appointment separately."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "old_appointment_id": {
                        "type": "integer",
                        "description": (
                            "The Cliniko individual appointment ID being replaced, from "
                            "get_patient_appointments."
                        ),
                    },
                    "first_name": {"type": "string", "description": "The caller's first name."},
                    "last_name": {"type": "string", "description": "The caller's last name."},
                    "phone_number": {"type": "string", "description": "The caller's phone number."},
                    "business_id": {
                        "type": "integer",
                        "description": (
                            "The Cliniko business ID for the new slot's branch — never guess."
                        ),
                    },
                    "practitioner_id": {
                        "type": "integer",
                        "description": "The Cliniko practitioner ID for the new slot — never guess.",
                    },
                    "appointment_type_id": {
                        "type": "integer",
                        "description": "The Cliniko appointment type ID for the new slot's service.",
                    },
                    "start_time": {
                        "type": "string",
                        "description": "ISO 8601 start time of the new slot.",
                    },
                    "end_time": {"type": "string", "description": "ISO 8601 end time of the new slot."},
                },
                "required": [
                    "old_appointment_id",
                    "first_name",
                    "last_name",
                    "phone_number",
                    "business_id",
                    "practitioner_id",
                    "appointment_type_id",
                    "start_time",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_appointment",
            "description": (
                "Cancel an existing individual appointment. When rescheduling, call this "
                "first on the old appointment before booking the new slot."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "appointment_id": {
                        "type": "integer",
                        "description": "The Cliniko individual appointment ID to cancel.",
                    },
                },
                "required": ["appointment_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_patient_appointments",
            "description": (
                "Look up a patient's existing appointments. Use this to find the "
                "appointment_id before cancelling or rescheduling a booking."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "integer",
                        "description": "The Cliniko patient ID whose appointments to look up.",
                    },
                },
                "required": ["patient_id"],
            },
        },
    },
]

ROLE_MAP = {"agent": "assistant", "user": "user"}


class LlmClient:
    """Drafts GPT-4o responses for Retell `response_required` events, executing Cliniko tools as needed."""

    def _build_messages(
        self, transcript: list[dict[str, Any]], clinic_context: str | None = None
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if clinic_context:
            messages.append({"role": "system", "content": clinic_context})
        for turn in transcript:
            role = ROLE_MAP.get(turn.get("role"), "user")
            messages.append({"role": role, "content": turn.get("content", "")})
        return messages

    async def _stream_and_forward(
        self,
        messages: list[dict[str, Any]],
        response_id: Any,
        websocket: Any,
    ) -> tuple[str, dict[int, dict[str, Any]]]:
        content_acc = ""
        tool_calls_acc: dict[int, dict[str, Any]] = {}

        stream = await aclient.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=TOOLS,
            stream=True,
        )

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            if delta.content:
                content_acc += delta.content
                await websocket.send_json(
                    {"response_id": response_id, "content": delta.content, "content_complete": False}
                )

            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    entry = tool_calls_acc.setdefault(
                        tool_call.index, {"id": None, "name": None, "arguments": ""}
                    )
                    if tool_call.id:
                        entry["id"] = tool_call.id
                    if tool_call.function and tool_call.function.name:
                        entry["name"] = tool_call.function.name
                    if tool_call.function and tool_call.function.arguments:
                        entry["arguments"] += tool_call.function.arguments

        return content_acc, tool_calls_acc

    async def _execute_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        start = time.monotonic()
        try:
            if name == "check_availability":
                available_times = await cliniko_client.get_available_times(
                    business_id=arguments["business_id"],
                    practitioner_id=arguments["practitioner_id"],
                    appointment_type_id=arguments["appointment_type_id"],
                    from_date=arguments["start_date"].split("T")[0],
                    to_date=arguments["end_date"].split("T")[0],
                )
                result = {"available_times": available_times}
            elif name == "book_appointment":
                result = await self._book_appointment(arguments)
            elif name == "reschedule_appointment":
                result = await self._reschedule_appointment(arguments)
            elif name == "cancel_appointment":
                success = await cliniko_client.cancel_appointment(arguments["appointment_id"])
                result = {"cancelled": success}
            elif name == "get_patient_appointments":
                result = await cliniko_client.get_patient_appointments(arguments["patient_id"])
            else:
                logger.warning("Unknown tool requested by model: %s", name)
                result = {"error": f"Unknown tool: {name}"}
        except ClinikoAPIError as exc:
            logger.error("Tool execution failed for %s: %s", name, exc)
            result = {"error": str(exc)}

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info("Executed tool=%s args=%s in %.1fms", name, arguments, elapsed_ms)
        return result

    async def _resolve_patient_id(self, first_name: str, last_name: str, phone_number: str) -> int:
        # Cliniko rejects filtering on phone_numbers.number ("not filterable"), so match on
        # name instead — this is only used to avoid creating duplicate patient records.
        existing_patients = await cliniko_client.get_patients_by_name(first_name, last_name)
        for patient in existing_patients:
            if (
                (patient.get("first_name") or "").strip().lower() == first_name.strip().lower()
                and (patient.get("last_name") or "").strip().lower() == last_name.strip().lower()
            ):
                logger.info(
                    "Matched existing patient_id=%s for %s %s at %s",
                    patient["id"],
                    first_name,
                    last_name,
                    phone_number,
                )
                return patient["id"]

        logger.info(
            "No existing patient match for %s %s at %s; creating new patient",
            first_name,
            last_name,
            phone_number,
        )
        return await cliniko_client.create_patient(first_name, last_name, phone_number)

    async def _compute_end_time(self, start_time: str, appointment_type_id: int) -> str:
        """Derive end_time from the service's real duration — never a guessed length."""
        appointment_types = await cliniko_client.get_appointment_types()
        duration = next(
            (t["duration_in_minutes"] for t in appointment_types if t["id"] == appointment_type_id),
            None,
        )
        if duration is None:
            raise ClinikoAPIError(f"Unknown appointment_type_id {appointment_type_id}")
        start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        return (start_dt + timedelta(minutes=duration)).isoformat().replace("+00:00", "Z")

    async def _book_appointment(self, arguments: dict[str, Any]) -> dict[str, Any]:
        business_id = arguments["business_id"]
        practitioner_id = arguments["practitioner_id"]
        appointment_type_id = arguments["appointment_type_id"]
        start_time = arguments["start_time"]
        end_time = arguments.get("end_time") or await self._compute_end_time(
            start_time, appointment_type_id
        )

        lock_acquired = await db.acquire_slot_lock(start_time, business_id, practitioner_id)
        if not lock_acquired:
            return {
                "error": (
                    "Race condition detected. This slot was just booked by another user. "
                    "Apologize and offer the next available slot."
                )
            }

        patient_id = await self._resolve_patient_id(
            arguments["first_name"], arguments["last_name"], arguments["phone_number"]
        )
        return await cliniko_client.book_appointment(
            patient_id=patient_id,
            practitioner_id=practitioner_id,
            appointment_type_id=appointment_type_id,
            business_id=business_id,
            start_time=start_time,
            end_time=end_time,
        )

    async def _reschedule_appointment(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Atomic reschedule: lock new slot -> book new -> cancel old only on success -> release lock.

        This ordering means a failure at any point never loses the caller's original
        appointment — it's only ever cancelled after the replacement is confirmed booked.
        """
        old_appointment_id = arguments["old_appointment_id"]
        business_id = arguments["business_id"]
        practitioner_id = arguments["practitioner_id"]
        appointment_type_id = arguments["appointment_type_id"]
        start_time = arguments["start_time"]
        end_time = arguments.get("end_time") or await self._compute_end_time(
            start_time, appointment_type_id
        )

        # 1. Lock the new slot.
        lock_acquired = await db.acquire_slot_lock(start_time, business_id, practitioner_id)
        if not lock_acquired:
            return {
                "error": (
                    "Race condition detected. The new slot was just booked by another user. "
                    "The caller's original appointment is untouched. Apologize and offer "
                    "another time."
                )
            }

        # 2. Book the new appointment before touching the old one.
        try:
            patient_id = await self._resolve_patient_id(
                arguments["first_name"], arguments["last_name"], arguments["phone_number"]
            )
            new_appointment = await cliniko_client.book_appointment(
                patient_id=patient_id,
                practitioner_id=practitioner_id,
                appointment_type_id=appointment_type_id,
                business_id=business_id,
                start_time=start_time,
                end_time=end_time,
            )
        except ClinikoAPIError as exc:
            await db.release_slot_lock(start_time, business_id, practitioner_id)
            logger.error(
                "Reschedule failed to book new slot; old appointment %s left intact: %s",
                old_appointment_id,
                exc,
            )
            return {
                "error": (
                    "Could not book the new slot, so the original appointment was left in "
                    "place and was NOT cancelled. Apologize and offer to try a different time."
                )
            }

        # 3. New slot is confirmed booked — only now cancel the old appointment.
        try:
            await cliniko_client.cancel_appointment(old_appointment_id)
        except ClinikoAPIError as exc:
            logger.error(
                "Reschedule booked new appointment but failed to cancel old appointment %s: %s",
                old_appointment_id,
                exc,
            )
            # 4. Release the lock regardless — Cliniko is now the source of truth for the
            # new slot, and holding the lock forever would block it from ever being reused.
            await db.release_slot_lock(start_time, business_id, practitioner_id)
            return {
                "new_appointment": new_appointment,
                "warning": (
                    "The new appointment was booked successfully, but the OLD appointment "
                    f"(id {old_appointment_id}) could NOT be cancelled automatically. Tell "
                    "the caller they currently have two appointments and that clinic staff "
                    "will follow up to cancel the old one."
                ),
            }

        # 4. Both steps succeeded — release the lock.
        await db.release_slot_lock(start_time, business_id, practitioner_id)
        return {
            "rescheduled": True,
            "new_appointment": new_appointment,
            "old_appointment_id": old_appointment_id,
        }

    def _finalize_tool_calls(self, tool_calls_acc: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "id": entry["id"],
                "type": "function",
                "function": {"name": entry["name"], "arguments": entry["arguments"]},
            }
            for _, entry in sorted(tool_calls_acc.items())
        ]

    async def draft_response(
        self, request: dict[str, Any], websocket: Any, clinic_context: str | None = None
    ) -> None:
        response_id = request.get("response_id")
        start = time.monotonic()
        logger.info("draft_response started: response_id=%s", response_id)

        try:
            transcript = request.get("transcript", [])
            messages = self._build_messages(transcript, clinic_context)

            content, tool_calls_acc = await self._stream_and_forward(messages, response_id, websocket)

            if tool_calls_acc:
                tool_calls = self._finalize_tool_calls(tool_calls_acc)
                messages.append(
                    {"role": "assistant", "content": content or None, "tool_calls": tool_calls}
                )

                for tool_call in tool_calls:
                    try:
                        arguments = json.loads(tool_call["function"]["arguments"] or "{}")
                    except json.JSONDecodeError:
                        logger.error(
                            "Failed to parse tool arguments for %s: %s",
                            tool_call["function"]["name"],
                            tool_call["function"]["arguments"],
                        )
                        arguments = {}

                    result = await self._execute_tool(tool_call["function"]["name"], arguments)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": json.dumps(result),
                        }
                    )

                await self._stream_and_forward(messages, response_id, websocket)

        except Exception:
            logger.exception("draft_response failed for response_id=%s", response_id)
            await websocket.send_json(
                {
                    "response_id": response_id,
                    "content": "I'm sorry, I'm having trouble right now. Let me get someone to help you.",
                    "content_complete": False,
                }
            )
        finally:
            await websocket.send_json({"response_id": response_id, "content": "", "content_complete": True})
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.info("draft_response completed: response_id=%s in %.1fms", response_id, elapsed_ms)


llm_client = LlmClient()
