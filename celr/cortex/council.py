"""
Hive-Mind Council — Phase 9: The Singularity Update
====================================================
ML Architect: Karpathy-style multi-model consensus.

A "Council of Experts" debates every high-stakes decision before the executor
acts on it. Three independent reviewers run IN PARALLEL (asyncio), then vote.

Workflow:
  1. Propose  → Nano-Cortex / heuristic generates an action/plan.
  2. Debate   → Three 'council members' critique it asynchronously.
  3. Consensus→ Majority vote determines the final action.
  4. Escalate → If split 1-1-1, escalate to a 'Chairman' model.

Speed Philosophy (Agno-inspired):
  - All three critics fire in parallel via asyncio.gather().
  - Total overhead ≈ time of ONE call, not three.
  - Gracefully degrades if a member fails (excluded from vote).
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

import litellm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class Verdict(str, Enum):
    APPROVE  = "APPROVE"
    REJECT   = "REJECT"
    ESCALATE = "ESCALATE"


@dataclass
class MemberVote:
    """A single council member's verdict with reasoning."""
    member_id: str
    model: str
    verdict: Verdict
    confidence: float   # 0.0 – 1.0
    reasoning: str
    tokens_used: int = 0


@dataclass
class CouncilDebate:
    """Full debate record for one decision checkpoint."""
    proposal: str
    votes: List[MemberVote] = field(default_factory=list)
    final_verdict: Optional[Verdict] = None
    final_reasoning: str = ""
    was_escalated: bool = False
    total_tokens: int = 0

    @property
    def approval_rate(self) -> float:
        if not self.votes:
            return 0.0
        approvals = sum(1 for v in self.votes if v.verdict == Verdict.APPROVE)
        return approvals / len(self.votes)

    @property
    def summary(self) -> str:
        parts = [f"Council Verdict: {self.final_verdict.value}"]
        parts.append(f"Approval: {self.approval_rate:.0%} ({len(self.votes)} members)")
        parts.append(f"Reasoning: {self.final_reasoning[:200]}")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Council Configuration
# ---------------------------------------------------------------------------

# Default council: diverse lightweight models (local-friendly via Ollama)
# Each member has a unique "persona" baked into the system prompt.
DEFAULT_COUNCIL: List[Dict] = [
    {
        "id": "Skeptic",
        "model": "ollama/llama3.2",
        "persona": (
            "You are a pragmatic skeptic. Your job is to find flaws in proposed plans. "
            "Be concise. Reply ONLY with: APPROVE, REJECT, or ESCALATE, then one sentence of reasoning."
        ),
    },
    {
        "id": "Optimist",
        "model": "ollama/llama3.2",  # Different temperature simulates diversity
        "persona": (
            "You are a creative optimist. You look for ways a plan could succeed. "
            "Be concise. Reply ONLY with: APPROVE, REJECT, or ESCALATE, then one sentence of reasoning."
        ),
    },
    {
        "id": "Realist",
        "model": "ollama/llama3.2",
        "persona": (
            "You are a data-driven realist. You evaluate plans based on facts and cost. "
            "Be concise. Reply ONLY with: APPROVE, REJECT, or ESCALATE, then one sentence of reasoning."
        ),
    },
]

CHAIRMAN_MODEL = "ollama/llama3.2"


# ---------------------------------------------------------------------------
# Async Member Call
# ---------------------------------------------------------------------------

async def _call_member(member: Dict, proposal: str) -> MemberVote:
    """
    Backend Wizard: fire a single async LLM call to one council member.
    Uses litellm.acompletion() for true async I/O (no thread blocking).
    """
    member_id = member["id"]
    model = member["model"]
    persona = member["persona"]

    messages = [
        {"role": "system", "content": persona},
        {"role": "user",   "content": f"Evaluate this proposed action:\n\n{proposal}"},
    ]

    try:
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            max_tokens=120,
            temperature=0.3,
        )
        raw_text = (response.choices[0].message.content or "").strip()
        tokens    = getattr(response.usage, "total_tokens", 0)

        # Parse verdict from first word
        first_word = raw_text.split()[0].upper() if raw_text else ""
        if first_word in ("APPROVE", "REJECT", "ESCALATE"):
            verdict    = Verdict(first_word)
            reasoning  = raw_text[len(first_word):].strip().lstrip(":").strip()
            confidence = 0.9
        else:
            # Heuristic fallback if model doesn't follow format
            if any(w in raw_text.upper() for w in ["APPROVE", "GOOD", "YES", "PROCEED"]):
                verdict, confidence = Verdict.APPROVE, 0.6
            elif any(w in raw_text.upper() for w in ["REJECT", "BAD", "NO", "STOP"]):
                verdict, confidence = Verdict.REJECT, 0.6
            else:
                verdict, confidence = Verdict.ESCALATE, 0.4
            reasoning = raw_text[:200]

        return MemberVote(
            member_id=member_id,
            model=model,
            verdict=verdict,
            confidence=confidence,
            reasoning=reasoning,
            tokens_used=tokens,
        )

    except Exception as e:
        logger.warning(f"Council member {member_id} failed: {e}. Abstaining.")
        return MemberVote(
            member_id=member_id,
            model=model,
            verdict=Verdict.ESCALATE,
            confidence=0.0,
            reasoning=f"Member unavailable: {e}",
        )


# ---------------------------------------------------------------------------
# The Council
# ---------------------------------------------------------------------------

class HiveMindCouncil:
    """
    Orchestrates the multi-model consensus debate.

    Usage:
        council = HiveMindCouncil()
        debate  = council.deliberate("Should we escalate step X?")
        if debate.final_verdict == Verdict.REJECT:
            # abort or retry
    """

    def __init__(
        self,
        members: Optional[List[Dict]] = None,
        chairman_model: str = CHAIRMAN_MODEL,
        quorum: int = 2,          # minimum votes needed for a majority
        approval_threshold: float = 0.6,
    ):
        self.members           = members or DEFAULT_COUNCIL
        self.chairman_model    = chairman_model
        self.quorum            = quorum
        self.approval_threshold = approval_threshold

    # ------------------------------------------------------------------
    # Public API  (sync wrapper so callers don't need to manage loops)
    # ------------------------------------------------------------------

    def deliberate(self, proposal: str) -> CouncilDebate:
        """
        Synchronous façade: run the async debate and return the result.
        Handles event-loop creation/reuse automatically.
        """
        try:
            loop = asyncio.get_running_loop()
            # We're inside an async context — schedule as a task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self._async_deliberate(proposal))
                return future.result(timeout=30)
        except RuntimeError:
            # No running loop — we're in a sync context, safe to call run
            return asyncio.run(self._async_deliberate(proposal))

    # ------------------------------------------------------------------
    # Async Core
    # ------------------------------------------------------------------

    async def _async_deliberate(self, proposal: str) -> CouncilDebate:
        debate = CouncilDebate(proposal=proposal)

        # ── Stage 1: Parallel Generation ───────────────────────────────
        tasks  = [_call_member(m, proposal) for m in self.members]
        votes  = await asyncio.gather(*tasks, return_exceptions=False)
        debate.votes = [v for v in votes if isinstance(v, MemberVote)]
        debate.total_tokens = sum(v.tokens_used for v in debate.votes)

        # ── Stage 2: Tally ─────────────────────────────────────────────
        tally: Dict[Verdict, int] = {v: 0 for v in Verdict}
        for vote in debate.votes:
            tally[vote.verdict] += 1

        logger.info(
            f"Council tally: APPROVE={tally[Verdict.APPROVE]} "
            f"REJECT={tally[Verdict.REJECT]} ESCALATE={tally[Verdict.ESCALATE]}"
        )

        # ── Stage 3: Majority Decision ─────────────────────────────────
        if tally[Verdict.APPROVE] >= self.quorum:
            debate.final_verdict  = Verdict.APPROVE
            debate.final_reasoning = "Majority approved."
        elif tally[Verdict.REJECT] >= self.quorum:
            debate.final_verdict  = Verdict.REJECT
            debate.final_reasoning = "Majority rejected."
        else:
            # Tie → Chairman's Decree
            debate.was_escalated  = True
            debate.final_verdict, debate.final_reasoning = await self._chairman_decree(debate)

        logger.info(f"Final council verdict: {debate.final_verdict.value}")
        return debate

    async def _chairman_decree(self, debate: CouncilDebate) -> Tuple[Verdict, str]:
        """
        Stage 3: The Chairman receives all votes and makes the final call.
        Inspired by Karpathy's 'Chairman's Decree' synthesis step.
        """
        votes_summary = "\n".join(
            f"- {v.member_id} ({v.verdict.value}): {v.reasoning}"
            for v in debate.votes
        )
        prompt = (
            f"The council is split on this proposal:\n\n"
            f"PROPOSAL: {debate.proposal}\n\n"
            f"VOTES:\n{votes_summary}\n\n"
            f"As Chairman, make the final decision. Reply with APPROVE or REJECT and one sentence."
        )
        try:
            response = await litellm.acompletion(
                model=self.chairman_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=80,
                temperature=0.1,
            )
            raw = (response.choices[0].message.content or "APPROVE").strip()
            first = raw.split()[0].upper()
            verdict   = Verdict.APPROVE if "APPROVE" in first else Verdict.REJECT
            reasoning = f"[Chairman] {raw[len(first):].strip()}"
            return verdict, reasoning
        except Exception as e:
            logger.error(f"Chairman call failed: {e}. Defaulting to APPROVE.")
            return Verdict.APPROVE, f"Chairman unavailable, defaulting to proceed."


# ---------------------------------------------------------------------------
# Convenience singleton
# ---------------------------------------------------------------------------

_default_council: Optional[HiveMindCouncil] = None


def get_council() -> HiveMindCouncil:
    """Returns the global singleton council instance."""
    global _default_council
    if _default_council is None:
        _default_council = HiveMindCouncil()
    return _default_council
