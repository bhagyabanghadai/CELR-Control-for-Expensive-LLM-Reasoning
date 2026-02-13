"""
Adaptive Cortex Router.

Decides whether a user query requires the full reasoning engine ("System 2")
or can be answered immediately by the LLM ("System 1").

Uses a fast rule-based pre-filter for obvious cases, then falls back to LLM
classification only when needed. This avoids wasting time on LLM calls for
simple greetings, factoids, and short questions.
"""

import re
from typing import Tuple, Dict, Any
from celr.core.llm import BaseLLMProvider
import json


# ── Rule-Based Pre-Filter Patterns ──────────────────────────────
# Greetings and social phrases (case-insensitive)
_GREETING_PATTERNS = [
    r"^(hi|hello|hey|howdy|yo|sup|hiya|good\s*(morning|afternoon|evening|night))[\s!?.]*$",
    r"^how\s+are\s+you",
    r"^what'?s?\s+up",
    r"^(thanks|thank\s+you|thx|bye|goodbye|see\s+you|take\s+care)",
    r"^(ok|okay|sure|got\s+it|cool|nice|great|awesome|alright)",
    r"^(yes|no|maybe|nah|yep|nope)[\s!?.]*$",
]

# Keywords that strongly signal complex reasoning
_REASONING_KEYWORDS = [
    "compare", "analyze", "explain step by step", "prove", "derive",
    "write a script", "write code", "implement", "build a",
    "debug", "fix the bug", "optimize", "refactor",
    "calculate", "solve", "what is the time complexity",
    "design a system", "architecture",
    "pro and con", "pros and cons", "trade-off", "trade off",
    "multi-step", "step by step",
]

_greeting_re = [re.compile(p, re.IGNORECASE) for p in _GREETING_PATTERNS]
_reasoning_keywords_lower = [k.lower() for k in _REASONING_KEYWORDS]


class Router:
    def __init__(self, llm: BaseLLMProvider):
        self.llm = llm

    @staticmethod
    def _fast_classify(prompt: str) -> Tuple[str, str] | None:
        """
        Rule-based pre-filter:  instant classification for obvious cases.
        
        Returns (category, reason) if confident, or None to defer to LLM.
        """
        text = prompt.strip()

        # Empty or very short → DIRECT
        if len(text) < 3:
            return "DIRECT", "Very short input"

        # Check greeting patterns
        for pat in _greeting_re:
            if pat.search(text):
                return "DIRECT", "Greeting/social phrase"

        # Word count heuristic: ≤ 6 words and no reasoning keywords → DIRECT
        words = text.split()
        text_lower = text.lower()

        # Check for strong reasoning signals
        for keyword in _reasoning_keywords_lower:
            if keyword in text_lower:
                return "REASONING", f"Contains reasoning keyword: '{keyword}'"

        # Short queries without reasoning keywords → DIRECT
        if len(words) <= 8:
            return "DIRECT", "Short query, no complexity signals"

        # Can't decide — defer to LLM
        return None

    def classify(self, prompt: str) -> Tuple[str, str]:
        """
        Determine if the task is 'simple' or 'complex'.
        
        Uses a fast rule-based pre-filter first, then falls back to LLM
        classification only for ambiguous queries.
        
        Returns:
            (category, reasoning)
            category: "DIRECT" or "REASONING"
        """
        # ── Fast path: rule-based ──
        fast_result = self._fast_classify(prompt)
        if fast_result is not None:
            return fast_result

        # ── Slow path: LLM-based (only for ambiguous queries) ──
        router_prompt = (
            f"Analyze the following user query: \"{prompt}\"\n\n"
            "Is this a complex reasoning task that requires a multi-step plan, math, or coding?\n"
            "Or is it a simple greeting, factoid, or conversation?\n\n"
            "Respond with JSON only: {{ \"type\": \"DIRECT\" or \"REASONING\", \"reason\": \"...\" }}"
        )

        try:
            response, _ = self.llm.generate(
                prompt=router_prompt,
                system_prompt="You are a classifier. Output JSON only."
            )
            
            cleaned = response.strip().replace("```json", "").replace("```", "")
            data = json.loads(cleaned)
            return data.get("type", "DIRECT"), data.get("reason", "")
            
        except Exception as e:
            return "DIRECT", f"Classification failed: {e}"
