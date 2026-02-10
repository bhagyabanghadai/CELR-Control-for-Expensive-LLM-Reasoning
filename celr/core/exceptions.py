"""
CELR Custom Exception Hierarchy

All CELR-specific exceptions inherit from CELRError.
This enables:
  - Catching ALL celr errors:      except CELRError
  - Catching specific categories:  except BudgetExhaustedError
  - Clean error messages with structured context
"""


class CELRError(Exception):
    """Base exception for all CELR errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


class BudgetExhaustedError(CELRError):
    """Raised when the cost budget is exceeded or a step can't be afforded."""

    def __init__(self, budget_limit: float, current_spend: float, step_description: str = ""):
        self.budget_limit = budget_limit
        self.current_spend = current_spend
        msg = (
            f"Budget exhausted: spent ${current_spend:.4f} / ${budget_limit:.4f} limit."
        )
        if step_description:
            msg += f" Cannot execute: '{step_description}'"
        super().__init__(msg, details={
            "budget_limit": budget_limit,
            "current_spend": current_spend,
            "step": step_description,
        })


class PlanningError(CELRError):
    """Raised when plan generation or parsing fails."""

    def __init__(self, message: str, raw_response: str = ""):
        self.raw_response = raw_response
        super().__init__(message, details={"raw_response": raw_response[:500]})


class EscalationError(CELRError):
    """Raised when model escalation/selection fails."""
    pass


class LLMProviderError(CELRError):
    """Raised when an LLM API call fails (after retries)."""

    def __init__(self, message: str, provider: str = "", model: str = ""):
        self.provider = provider
        self.model = model
        super().__init__(message, details={"provider": provider, "model": model})


class ToolExecutionError(CELRError):
    """Raised when a tool (Python REPL, shell, etc.) fails."""

    def __init__(self, message: str, tool_name: str = "", code: str = ""):
        self.tool_name = tool_name
        super().__init__(message, details={
            "tool_name": tool_name,
            "code": code[:200],
        })


class VerificationError(CELRError):
    """Raised when step verification fails after all retries."""

    def __init__(self, message: str, step_id: str = "", verification_notes: str = ""):
        self.step_id = step_id
        super().__init__(message, details={
            "step_id": step_id,
            "verification_notes": verification_notes[:300],
        })
