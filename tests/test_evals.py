"""Automated evaluation harness for the Retell/GPT-4o agent's prompt behavior.

These are *evals*, not unit tests: they make real calls to Groq's
llama-3.3-70b-versatile via LlmClient.draft_response to prove the live
prompt + tool-schema combination still behaves correctly before it ships.
A real GROQ_API_KEY is required (see tests/conftest.py, which currently
seeds a dummy value that will cause these evals to fail auth against the
live API unless overridden). Cliniko/DB access is stubbed out by patching
LlmClient._execute_tool, so we can inspect exactly which tool the model
chose to call and with what arguments, without needing live Cliniko creds.
"""

import asyncio
import json

import pytest

from app.services.llm import LlmClient

EVAL_TIMEOUT_SECONDS = 30


class MockWebSocket:
    """Stands in for the Retell websocket connection."""

    def __init__(self) -> None:
        self.emitted_messages: list[dict] = []

    async def send_json(self, data: dict) -> None:
        self.emitted_messages.append(data)


def assembled_text(mock_ws: MockWebSocket) -> str:
    return "".join(message.get("content", "") for message in mock_ws.emitted_messages)


async def run_draft_response(llm_client: LlmClient, *args, **kwargs) -> None:
    await asyncio.wait_for(
        llm_client.draft_response(*args, **kwargs), timeout=EVAL_TIMEOUT_SECONDS
    )


@pytest.mark.asyncio
async def test_eval_cross_branch_search(monkeypatch):
    llm_client = LlmClient()
    mock_ws = MockWebSocket()

    captured_tool_calls: list[tuple[str, dict]] = []

    async def fake_execute_tool(name: str, arguments: dict) -> dict:
        captured_tool_calls.append((name, arguments))
        return {
            "available_slots": [
                {"business_id": 2, "practitioner_id": 5, "start_time": "2026-07-18T09:00:00Z"}
            ]
        }

    monkeypatch.setattr(llm_client, "_execute_tool", fake_execute_tool)

    request = {
        "response_id": "eval-1",
        "transcript": [
            {
                "role": "user",
                "content": (
                    "I need the earliest available appointment you have. "
                    "I don't care which clinic."
                ),
            }
        ],
    }

    await run_draft_response(
        llm_client,
        request,
        mock_ws,
        clinic_context="Branches: ID 1 (Downtown), ID 2 (Uptown)",
    )

    assert captured_tool_calls, (
        f"Expected the model to call a tool, but none was called. "
        f"Assembled response: {assembled_text(mock_ws)!r}"
    )

    tool_name, tool_args = captured_tool_calls[0]
    assert tool_name == "check_availability"
    assert "business_id" not in tool_args, (
        "business_id should be omitted to search across all branches, "
        f"but got arguments: {json.dumps(tool_args)}"
    )


@pytest.mark.asyncio
async def test_eval_code_switching(monkeypatch):
    llm_client = LlmClient()
    mock_ws = MockWebSocket()

    captured_tool_calls: list[tuple[str, dict]] = []

    async def fake_execute_tool(name: str, arguments: dict) -> dict:
        captured_tool_calls.append((name, arguments))
        return {
            "available_slots": [
                {"business_id": 1, "practitioner_id": 3, "start_time": "2026-07-18T14:00:00Z"}
            ]
        }

    monkeypatch.setattr(llm_client, "_execute_tool", fake_execute_tool)

    request = {
        "response_id": "eval-2",
        "transcript": [
            {
                "role": "user",
                "content": (
                    "Hi, mera naam Rahul hai, can I get an appointment for tomorrow afternoon?"
                ),
            }
        ],
    }

    await run_draft_response(
        llm_client,
        request,
        mock_ws,
        clinic_context=(
            "Today's date is 2026-07-17. Rahul is a returning patient at this clinic."
        ),
    )

    text = assembled_text(mock_ws).lower()
    hinglish_markers = ["haan", "theek", "zaroor", "bilkul", "ji", "accha", "aap", "namaste"]
    assert any(marker in text for marker in hinglish_markers), (
        "Expected the agent to code-switch into Hindi/Hinglish vocabulary to match "
        f"the caller, but got a purely English-sounding response: {text!r}"
    )

    assert captured_tool_calls, (
        f"Expected the model to call check_availability. Assembled response: {text!r}"
    )
    tool_name, tool_args = captured_tool_calls[0]
    assert tool_name == "check_availability"
    assert tool_args.get("time_preference", "").lower() == "afternoon"
    assert "2026-07-18" in tool_args.get("start_date", ""), (
        f"Expected the search window to target tomorrow (2026-07-18), got: {json.dumps(tool_args)}"
    )
