import logging

from celr.core.types import Step, TaskContext
from celr.core.llm import BaseLLMProvider

logger = logging.getLogger(__name__)


class SelfReflection:
    """
    Reflexion-style failure analysis engine.

    Analyzes failed steps and suggests fixes. Decides whether to retry
    based on error type, attempt count, and cost analysis.
    """

    def __init__(self, llm: BaseLLMProvider, max_retries: int = 3):
        self.llm = llm
        self.max_retries = max_retries
        self.total_reflection_cost = 0.0

    def analyze_failure(self, step: Step, context: TaskContext) -> str:
        """
        Analyzes a failed step and suggests a fix or a better approach.
        """
        prompt = f"""FAILURE ANALYSIS

The following step failed:
Description: {step.description}
Error/Notes: {step.verification_notes or step.output}

Context History (recent):
{context.execution_history[-3:]}

Analyze WHY it failed and suggest a SPECIFIC fix.
Be concise — one paragraph max.
"""

        analysis, usage = self.llm.generate(prompt)
        cost = self.llm.calculate_cost(usage)
        self.total_reflection_cost += cost

        context.log(f"Reflection on Step {step.id}: {analysis[:200]}")
        logger.info(f"Reflection analysis for step {step.id} ({usage.total_tokens} tokens, ${cost:.6f})")
        return analysis

    def should_retry(self, step: Step, attempt_count: int) -> bool:
        """
        Decides if we should retry based on error type and attempt count.

        Smart retry logic:
          - Always retry on rate limits (transient)
          - Retry on logic/verification errors (with reflection)
          - Don't retry on budget exhaustion
          - Don't retry on safety violations
          - Respect max_retries limit
        """
        if attempt_count >= self.max_retries:
            logger.info(f"Step {step.id}: max retries ({self.max_retries}) reached, giving up")
            return False

        error_text = (step.verification_notes or step.output or "").lower()

        # Never retry on these
        no_retry_patterns = [
            "budget exhausted",
            "budget exceeded",
            "unsafe",
            "permission denied",
            "authentication failed",
            "api key",
        ]
        for pattern in no_retry_patterns:
            if pattern in error_text:
                logger.info(f"Step {step.id}: non-retryable error pattern '{pattern}'")
                return False

        # Always retry on these (transient errors)
        always_retry_patterns = [
            "rate limit",
            "rate_limit",
            "too many requests",
            "timeout",
            "connection",
            "503",
            "502",
            "504",
        ]
        for pattern in always_retry_patterns:
            if pattern in error_text:
                logger.info(f"Step {step.id}: transient error '{pattern}', retrying")
                return True

        # Default: retry (with reflection providing a fix)
        return True
