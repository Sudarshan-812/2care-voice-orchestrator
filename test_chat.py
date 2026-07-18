"""Interactive terminal chat for manually exercising the voice agent's LLM logic.

Runs the real LlmClient against the real Groq-backed model and real Cliniko
tool execution (booking, cancelling, rescheduling), driven by typed input
instead of a live Retell call. Useful for manually walking through a
conversation end-to-end before shipping.

Usage:
    python test_chat.py
"""

import asyncio
from datetime import datetime
from typing import Any

from app.services.cliniko import ClinikoAPIError, cliniko_client
from app.services.llm import LlmClient


class MockWebSocket:
    """Stands in for the Retell websocket: prints streamed text live and
    records every tool call the agent makes along the way."""

    def __init__(self) -> None:
        self.buffer = ""
        self.tool_calls: list[dict[str, Any]] = []

    async def send_json(self, data: dict) -> None:
        if data.get("type") == "response_audio_chunk":
            text = data.get("content", "") or data.get("text", "")
        else:
            text = data.get("content", "")

        if text:
            self.buffer += text
            print(text, end="", flush=True)

    def record_tool_call(self, name: str, arguments: dict, result: Any) -> None:
        self.tool_calls.append({"name": name, "arguments": arguments, "result": result})
        print(f"\n[tool call] {name}({arguments})")
        print(f"[tool result] {result}\n")

    def reset_turn(self) -> None:
        self.buffer = ""


async def build_clinic_context() -> str:
    """Best-effort mirror of the context websocket.py builds for a real call."""
    context_parts = [f"Today is {datetime.now().strftime('%A, %B %d, %Y, %I:%M %p')}."]

    try:
        appointment_types = await cliniko_client.get_appointment_types()
    except ClinikoAPIError as exc:
        print(f"[warning] Could not load appointment types from Cliniko: {exc}")
        appointment_types = []
    if appointment_types:
        services = ", ".join(
            f"ID {t['id']} ({t['name']}, {t['duration_in_minutes']} mins)" for t in appointment_types
        )
        context_parts.append(f"The clinic offers the following services: {services}.")

    try:
        businesses = await cliniko_client.get_businesses()
    except ClinikoAPIError as exc:
        print(f"[warning] Could not load branches from Cliniko: {exc}")
        businesses = []
    if businesses:
        branches = ", ".join(f"ID {b['id']} ({b['name']})" for b in businesses)
        context_parts.append(f"The clinic has the following locations (branches): {branches}.")

    return " ".join(context_parts)


async def main() -> None:
    llm_client = LlmClient()
    mock_ws = MockWebSocket()

    original_execute_tool = llm_client._execute_tool

    async def logging_execute_tool(name: str, arguments: dict) -> Any:
        result = await original_execute_tool(name, arguments)
        mock_ws.record_tool_call(name, arguments, result)
        return result

    llm_client._execute_tool = logging_execute_tool

    clinic_context = await build_clinic_context()
    print("2care.ai voice agent — text chat harness. Type 'exit' to quit.\n")

    transcript: list[dict[str, str]] = []
    response_id = 0

    while True:
        user_input = input("\nYou: ")
        if user_input.strip().lower() == "exit":
            break

        transcript.append({"role": "user", "content": user_input})
        response_id += 1

        request = {"response_id": response_id, "transcript": list(transcript)}

        mock_ws.reset_turn()
        print("Agent: ", end="", flush=True)
        await llm_client.draft_response(request, mock_ws, clinic_context=clinic_context)
        print()

        transcript.append({"role": "agent", "content": mock_ws.buffer})


if __name__ == "__main__":
    asyncio.run(main())
