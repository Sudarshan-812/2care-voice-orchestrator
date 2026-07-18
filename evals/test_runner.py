"""Tier 1 — text-level scenario suite (CI, fast).

Drives the real LlmClient.draft_response for each scenario in scenarios.yaml,
bypassing the Retell/telephony layer entirely and stubbing out real Cliniko/
Postgres calls via LlmClient._execute_tool, then asserts which tools were
called, in what order, and which must NOT have been called.

This still makes real calls to Groq (only tool *execution* is faked), so it
needs a real GROQ_API_KEY — see conftest.py at the repo root, which loads
.env and only falls back to a dummy key if none is found.
"""

import asyncio
from pathlib import Path
from typing import Any

import pytest
import yaml

from app.services.llm import LlmClient

SCENARIOS_PATH = Path(__file__).parent / "scenarios.yaml"
EVAL_TIMEOUT_SECONDS = 30

with SCENARIOS_PATH.open(encoding="utf-8") as f:
    SCENARIOS: list[dict[str, Any]] = yaml.safe_load(f)


class MockWebSocket:
    """Stands in for the Retell websocket: just accumulates the assembled reply text."""

    def __init__(self) -> None:
        self.buffer = ""

    async def send_json(self, data: dict) -> None:
        content = data.get("content", "")
        if content:
            self.buffer += content

    def reset_turn(self) -> None:
        self.buffer = ""


def is_subsequence(expected: list[str], actual: list[str]) -> bool:
    """True if `expected` names appear in `actual`, in order (extra calls allowed between)."""
    it = iter(actual)
    return all(name in it for name in expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["name"] for s in SCENARIOS])
async def test_scenario(scenario: dict[str, Any]) -> None:
    llm_client = LlmClient()
    mock_ws = MockWebSocket()

    captured_tool_names: list[str] = []
    mock_tool_results: dict[str, Any] = scenario.get("mock_tool_results") or {}

    async def fake_execute_tool(name: str, arguments: dict) -> dict:
        captured_tool_names.append(name)
        return mock_tool_results.get(name, {})

    llm_client._execute_tool = fake_execute_tool

    transcript: list[dict[str, str]] = []
    clinic_context = scenario["clinic_context"]
    last_reply = ""

    for turn_index, user_text in enumerate(scenario["user_turns"]):
        transcript.append({"role": "user", "content": user_text})
        mock_ws.reset_turn()

        request = {
            "response_id": f"{scenario['name']}-{turn_index}",
            "transcript": list(transcript),
        }
        await asyncio.wait_for(
            llm_client.draft_response(request, mock_ws, clinic_context=clinic_context),
            timeout=EVAL_TIMEOUT_SECONDS,
        )

        last_reply = mock_ws.buffer
        transcript.append({"role": "agent", "content": last_reply})

    expected_sequence = scenario.get("expected_tool_sequence") or []
    forbidden_tools = scenario.get("forbidden_tools") or []

    assert is_subsequence(expected_sequence, captured_tool_names), (
        f"[{scenario['name']}] expected tool sequence {expected_sequence} to appear (in order) "
        f"in actual calls {captured_tool_names}. Final reply: {last_reply!r}"
    )

    called_forbidden = [name for name in captured_tool_names if name in forbidden_tools]
    assert not called_forbidden, (
        f"[{scenario['name']}] forbidden tool(s) were called: {called_forbidden} "
        f"(full call sequence: {captured_tool_names}). Final reply: {last_reply!r}"
    )

    content_checks = scenario.get("content_checks") or {}
    any_of = content_checks.get("any_of")
    if any_of:
        reply_lower = last_reply.lower()
        assert any(marker.lower() in reply_lower for marker in any_of), (
            f"[{scenario['name']}] expected the final reply to contain one of {any_of}, "
            f"but got: {last_reply!r}"
        )
