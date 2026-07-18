"""Interactive terminal chat for manually exercising the voice agent's LLM logic.

Runs the real LlmClient against the real Groq-backed model and real Cliniko
tool execution (booking, cancelling, rescheduling), driven by typed input
instead of a live Retell call. Useful for manually walking through a
conversation end-to-end before shipping.

Usage:
    python test_chat.py
"""

import asyncio
from typing import Any

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


async def main() -> None:
    llm_client = LlmClient()
    mock_ws = MockWebSocket()

    original_execute_tool = llm_client._execute_tool

    async def logging_execute_tool(name: str, arguments: dict) -> Any:
        result = await original_execute_tool(name, arguments)
        mock_ws.record_tool_call(name, arguments, result)
        return result

    llm_client._execute_tool = logging_execute_tool

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
        await llm_client.draft_response(
            request,
            mock_ws,
            clinic_context="Today's date is 2026-07-18. Available branches: ID 1 (Downtown Clinic), "
            "ID 2 (Uptown Clinic).",
        )
        print()

        transcript.append({"role": "agent", "content": mock_ws.buffer})


if __name__ == "__main__":
    asyncio.run(main())
