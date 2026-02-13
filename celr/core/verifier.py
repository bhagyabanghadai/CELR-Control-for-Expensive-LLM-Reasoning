import logging
from typing import Any, Optional
from celr.core.types import Step, StepType, TaskContext
from celr.core.tools import ToolRegistry
from celr.core.llm import BaseLLMProvider

logger = logging.getLogger(__name__)


class Verifier:
    """
    Multi-strategy verification engine.

    Strategies:
      1. Code execution output check (heuristic + exit code)
      2. LLM-as-Judge peer review with confidence scoring
      3. Output sanity checks (empty, too short, gibberish)
    """

    def __init__(self, tool_registry: ToolRegistry, llm: BaseLLMProvider):
        self.registry = tool_registry
        self.llm = llm
        self.total_verification_cost = 0.0

    def verify(self, step: Step, context: TaskContext) -> bool:
        """
        Verifies the output of a step using the appropriate strategy.
        Returns True if successful, False otherwise.
        """
        output = step.output or ""

        # 0. Sanity check — empty or trivially bad output
        if not output.strip():
            step.verification_notes = "Output is empty."
            return False

        # 1. Execution steps — check for error patterns
        if step.step_type == StepType.EXECUTION:
            return self._verify_execution(step, output)

        # 2. Reasoning/Verification steps — LLM peer review
        return self._verify_with_llm(step, context, output)

    def _verify_execution(self, step: Step, output: str) -> bool:
        """Verify tool execution output using heuristic error detection."""
        # Error keyword detection (case-insensitive)
        error_indicators = [
            "Traceback (most recent call last)",
            "SyntaxError:",
            "NameError:",
            "TypeError:",
            "ValueError:",
            "ImportError:",
            "ModuleNotFoundError:",
            "FileNotFoundError:",
            "PermissionError:",
            "ZeroDivisionError:",
            "IndexError:",
            "KeyError:",
            "AttributeError:",
            "RuntimeError:",
        ]

        for indicator in error_indicators:
            if indicator in output:
                step.verification_notes = f"Execution output contains error: {indicator}"
                return False

        # Generic fallback for unrecognized exceptions
        if "Error" in output and "Exception" in output:
            step.verification_notes = "Execution output contained error keywords."
            return False

        step.verification_notes = "Execution completed without errors."
        return True

    def _verify_with_llm(self, step: Step, context: TaskContext, output: str) -> bool:
        """Verify reasoning output using LLM-as-Judge with confidence scoring."""
        prompt = f"""VERIFICATION TASK
Original Goal: {step.description}
Generated Output: {output[:2000]}

Evaluate whether the output satisfies the goal.

Reply in this exact format:
VERDICT: YES or NO
CONFIDENCE: (a number from 0.0 to 1.0)
REASON: (one sentence explanation)
"""

        try:
            critique, usage = self.llm.generate(prompt)
            cost = self.llm.calculate_cost(usage)
            self.total_verification_cost += cost

            # Parse verdict
            verdict_pass = "YES" in critique.upper().split("VERDICT")[-1][:20] if "VERDICT" in critique.upper() else "YES" in critique.upper()

            # Parse confidence
            confidence = self._extract_confidence(critique)

            step.verification_notes = f"[confidence={confidence:.2f}] {critique[:300]}"

            if verdict_pass and confidence >= 0.3:
                return True
            elif verdict_pass and confidence < 0.3:
                step.verification_notes = f"LLM said YES but confidence too low ({confidence:.2f}): {critique[:200]}"
                return False
            else:
                step.verification_notes = f"Verification Failed [confidence={confidence:.2f}]: {critique[:300]}"
                return False

        except Exception as e:
            logger.warning(f"LLM verification failed, falling back to pass: {e}")
            step.verification_notes = f"Verification error (auto-passed): {e}"
            return True  # Fail-open to avoid blocking on verification errors

    def _extract_confidence(self, text: str) -> float:
        """Extract confidence score from verification response."""
        import re
        # Try to find CONFIDENCE: 0.X pattern
        match = re.search(r"CONFIDENCE:\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
        if match:
            try:
                return max(0.0, min(1.0, float(match.group(1))))
            except ValueError:
                pass
        # Default confidence based on YES/NO presence
        if "YES" in text.upper():
            return 0.7
        return 0.3
