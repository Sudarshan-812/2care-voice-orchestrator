"""Automated evaluation harness for the Retell/GPT-4o agent's prompt behavior.

These are *evals*, not unit tests: they make real calls to Groq's
llama-3.3-70b-versatile via LlmClient.draft_response to prove the live
prompt + tool-schema combination still behaves correctly before it ships.
A real GROQ_API_KEY is required (see tests/conftest.py, which currently
seeds a dummy value that will cause these evals to fail auth against the
live API unless overridden). Cliniko/DB access is stubbed out by patching
LlmClient._execute_tool, so we can inspect exactly which tool the model
chose to call and with what arguments, without needing live Cliniko creds.

business_id/practitioner_id/appointment_type_id are all strictly required
on check_availability now (see app/services/llm.py) — there is no "search
all branches" shorthand and no silent ID defaulting. Both evals' caller
utterances name a specific branch/service so the model has everything it
needs to act in a single turn, since this harness only drives one turn.
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from app.services.llm import LlmClient

EVAL_TIMEOUT_SECONDS = 30

CLINIC_CONTEXT = (
    "Branches: ID 1 (Downtown), ID 2 (Uptown). "
    "Practitioners: ID 10 (Dr. Asha Rao). "
    "Services: ID 20 (General Consultation, 30 mins)."
)


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
        return {"available_times": [{"appointment_start": "2026-07-19T09:00:00+00:00"}]}

    monkeypatch.setattr(llm_client, "_execute_tool", fake_execute_tool)

    request = {
        "response_id": "eval-1",
        "transcript": [
            {
                "role": "user",
                "content": (
                    "I need the earliest available general consultation you have. "
                    "I don't care which clinic — check every branch."
                ),
            }
        ],
    }

    await run_draft_response(llm_client, request, mock_ws, clinic_context=CLINIC_CONTEXT)

    availability_calls = [args for name, args in captured_tool_calls if name == "check_availability"]
    assert availability_calls, (
        f"Expected the model to call check_availability, but none was called. "
        f"Assembled response: {assembled_text(mock_ws)!r}"
    )

    checked_business_ids = {call.get("business_id") for call in availability_calls}
    assert checked_business_ids == {1, 2}, (
        "business_id is now required on check_availability — there's no 'search all "
        "branches' shorthand, so the model must call it once per known branch. Expected "
        f"business_ids {{1, 2}}, got {checked_business_ids}. Calls: {json.dumps(availability_calls)}"
    )

    for call in availability_calls:
        assert call.get("practitioner_id") is not None, f"Missing practitioner_id: {json.dumps(call)}"
        assert call.get("appointment_type_id") is not None, (
            f"Missing appointment_type_id: {json.dumps(call)}"
        )


@pytest.mark.asyncio
async def test_eval_code_switching(monkeypatch):
    llm_client = LlmClient()
    mock_ws = MockWebSocket()

    captured_tool_calls: list[tuple[str, dict]] = []
    today = datetime.now(timezone.utc)
    tomorrow = (today + timedelta(days=1)).strftime("%Y-%m-%d")

    async def fake_execute_tool(name: str, arguments: dict) -> dict:
        captured_tool_calls.append((name, arguments))
        return {"available_times": [{"appointment_start": f"{tomorrow}T14:00:00+00:00"}]}

    monkeypatch.setattr(llm_client, "_execute_tool", fake_execute_tool)

    request = {
        "response_id": "eval-2",
        "transcript": [
            {
                "role": "user",
                "content": (
                    "Hi, mera naam Rahul hai, mujhe general consultation ke liye kal "
                    "afternoon ka appointment chahiye Downtown clinic mein."
                ),
            }
        ],
    }

    await run_draft_response(
        llm_client,
        request,
        mock_ws,
        clinic_context=(
            f"Today's date is {today.strftime('%A, %B %d, %Y')}. "
            f"Rahul is a returning patient at this clinic. {CLINIC_CONTEXT}"
        ),
    )

    text = assembled_text(mock_ws).lower()
    hinglish_markers = ["haan", "theek", "zaroor", "bilkul", "ji", "accha", "aap", "namaste"]
    assert any(marker in text for marker in hinglish_markers), (
        "Expected the agent to code-switch into Hindi/Hinglish vocabulary to match "
        f"the caller, but got a purely English-sounding response: {text!r}"
    )

    availability_calls = [args for name, args in captured_tool_calls if name == "check_availability"]
    assert availability_calls, (
        f"Expected the model to call check_availability. Assembled response: {text!r}"
    )
    tool_args = availability_calls[0]
    assert tool_args.get("time_preference", "").lower() == "afternoon"
    assert tool_args.get("business_id") == 1
    assert tool_args.get("practitioner_id") is not None
    assert tool_args.get("appointment_type_id") is not None
    assert tomorrow in tool_args.get("start_date", ""), (
        f"Expected the search window to target tomorrow ({tomorrow}), got: {json.dumps(tool_args)}"
    )
