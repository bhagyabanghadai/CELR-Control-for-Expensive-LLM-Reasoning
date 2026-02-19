"""
Tests for the Hive-Mind Council (Phase 9).
Test Engineer: `python-testing-patterns` skill.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from celr.cortex.council import (
    HiveMindCouncil, MemberVote, CouncilDebate, Verdict,
    DEFAULT_COUNCIL, _call_member, get_council,
)


# ---------------------------------------------------------------------------
# Unit tests: MemberVote parsing
# ---------------------------------------------------------------------------

class TestMemberVote:
    def test_verdict_values(self):
        assert Verdict.APPROVE == "APPROVE"
        assert Verdict.REJECT  == "REJECT"
        assert Verdict.ESCALATE == "ESCALATE"

    def test_debate_approval_rate(self):
        debate = CouncilDebate(proposal="test")
        debate.votes = [
            MemberVote("A", "m1", Verdict.APPROVE,  0.9, "good"),
            MemberVote("B", "m2", Verdict.REJECT,   0.9, "bad"),
            MemberVote("C", "m3", Verdict.APPROVE,  0.9, "good"),
        ]
        assert debate.approval_rate == pytest.approx(2/3)

    def test_debate_summary_includes_verdict(self):
        debate = CouncilDebate(proposal="test")
        debate.final_verdict  = Verdict.APPROVE
        debate.final_reasoning = "Majority approved."
        debate.votes = []
        assert "APPROVE" in debate.summary


# ---------------------------------------------------------------------------
# Integration: async member call (mocked)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_call_member_approve():
    """Member correctly parses APPROVE response."""
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = "APPROVE looks good to me"
    mock_resp.usage.total_tokens = 42

    with patch("celr.cortex.council.litellm.acompletion", new=AsyncMock(return_value=mock_resp)):
        member = DEFAULT_COUNCIL[0]
        vote = await _call_member(member, "Should we escalate?")

    assert vote.verdict == Verdict.APPROVE
    assert "good" in vote.reasoning.lower()
    assert vote.tokens_used == 42


@pytest.mark.asyncio
async def test_call_member_reject():
    """Member correctly parses REJECT response."""
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = "REJECT this is risky"
    mock_resp.usage.total_tokens = 30

    with patch("celr.cortex.council.litellm.acompletion", new=AsyncMock(return_value=mock_resp)):
        vote = await _call_member(DEFAULT_COUNCIL[1], "Escalate?")

    assert vote.verdict == Verdict.REJECT


@pytest.mark.asyncio
async def test_call_member_failure_graceful():
    """Member failure results in ESCALATE abstain, not a crash."""
    with patch("celr.cortex.council.litellm.acompletion", side_effect=RuntimeError("timeout")):
        vote = await _call_member(DEFAULT_COUNCIL[2], "Escalate?")

    assert vote.verdict == Verdict.ESCALATE
    assert vote.confidence == 0.0


# ---------------------------------------------------------------------------
# Integration: Full council deliberation (mocked)
# ---------------------------------------------------------------------------

def _build_mock_votes(verdicts):
    """Helper: build mock acompletion responses for given verdicts."""
    resps = []
    for v in verdicts:
        m = MagicMock()
        m.choices[0].message.content = f"{v} because reasons"
        m.usage.total_tokens = 10
        resps.append(m)
    return resps


@pytest.mark.asyncio
async def test_council_majority_approve():
    """2/3 APPROVE → final verdict is APPROVE."""
    council = HiveMindCouncil()
    responses = _build_mock_votes(["APPROVE", "APPROVE", "REJECT"])

    with patch("celr.cortex.council.litellm.acompletion", new=AsyncMock(side_effect=responses)):
        debate = await council._async_deliberate("Escalate step X?")

    assert debate.final_verdict == Verdict.APPROVE
    assert len(debate.votes) == 3


@pytest.mark.asyncio
async def test_council_majority_reject():
    """2/3 REJECT → final verdict is REJECT."""
    council = HiveMindCouncil()
    responses = _build_mock_votes(["REJECT", "REJECT", "APPROVE"])

    with patch("celr.cortex.council.litellm.acompletion", new=AsyncMock(side_effect=responses)):
        debate = await council._async_deliberate("Escalate step X?")

    assert debate.final_verdict == Verdict.REJECT


@pytest.mark.asyncio
async def test_council_tie_triggers_chairman():
    """1/3/APPROVE, 1 REJECT, 1 ESCALATE → chairman triggered."""
    council = HiveMindCouncil()
    # 3 member calls, then 1 chairman call
    all_responses = _build_mock_votes(["APPROVE", "REJECT", "ESCALATE", "APPROVE"])

    with patch("celr.cortex.council.litellm.acompletion", new=AsyncMock(side_effect=all_responses)):
        debate = await council._async_deliberate("Escalate step X?")

    assert debate.was_escalated is True
    assert debate.final_verdict in (Verdict.APPROVE, Verdict.REJECT)


# ---------------------------------------------------------------------------
# Speed
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_council_parallel_speed():
    """All three members should complete in roughly one call's time (mocked)."""
    import time

    call_duration = 0.05  # 50ms simulated

    async def slow_response(*args, **kwargs):
        await asyncio.sleep(call_duration)
        m = MagicMock()
        m.choices[0].message.content = "APPROVE parallel!"
        m.usage.total_tokens = 5
        return m

    council = HiveMindCouncil()
    start = time.monotonic()

    with patch("celr.cortex.council.litellm.acompletion", new=slow_response):
        debate = await council._async_deliberate("Speed test?")

    elapsed = time.monotonic() - start
    # Should finish in ~1x call time, NOT 3x (serial would be ~0.15s)
    assert elapsed < call_duration * 2.5, (
        f"Parallel calls too slow: {elapsed:.3f}s (expected < {call_duration * 2.5:.3f}s)"
    )
    assert debate.final_verdict == Verdict.APPROVE


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

def test_get_council_singleton():
    """get_council() always returns the same instance."""
    c1 = get_council()
    c2 = get_council()
    assert c1 is c2
