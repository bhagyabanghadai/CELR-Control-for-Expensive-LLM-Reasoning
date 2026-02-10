"""Tests for celr.core.exceptions — Custom exception hierarchy."""

import pytest
from celr.core.exceptions import (
    CELRError,
    BudgetExhaustedError,
    PlanningError,
    EscalationError,
    LLMProviderError,
    ToolExecutionError,
    VerificationError,
)


class TestExceptionHierarchy:
    def test_all_inherit_from_celr_error(self):
        """All custom exceptions should inherit from CELRError."""
        assert issubclass(BudgetExhaustedError, CELRError)
        assert issubclass(PlanningError, CELRError)
        assert issubclass(EscalationError, CELRError)
        assert issubclass(LLMProviderError, CELRError)
        assert issubclass(ToolExecutionError, CELRError)
        assert issubclass(VerificationError, CELRError)

    def test_catch_all_celr_errors(self):
        """Should be able to catch all CELR errors with one except clause."""
        for exc_class in [BudgetExhaustedError, PlanningError, LLMProviderError]:
            with pytest.raises(CELRError):
                if exc_class == BudgetExhaustedError:
                    raise exc_class(budget_limit=1.0, current_spend=1.5)
                elif exc_class == PlanningError:
                    raise exc_class(message="test")
                else:
                    raise exc_class(message="test")


class TestBudgetExhaustedError:
    def test_message_format(self):
        e = BudgetExhaustedError(budget_limit=1.0, current_spend=1.5, step_description="Test step")
        assert "1.0000" in str(e)
        assert "1.5000" in str(e)
        assert "Test step" in str(e)

    def test_details(self):
        e = BudgetExhaustedError(budget_limit=1.0, current_spend=1.5)
        assert e.details["budget_limit"] == 1.0
        assert e.details["current_spend"] == 1.5


class TestToolExecutionError:
    def test_code_truncated(self):
        long_code = "x = 1\n" * 100
        e = ToolExecutionError(message="Error", tool_name="python_repl", code=long_code)
        assert len(e.details["code"]) <= 200


class TestPlanningError:
    def test_raw_response_truncated(self):
        long_response = "a" * 1000
        e = PlanningError(message="Parse failed", raw_response=long_response)
        assert len(e.details["raw_response"]) <= 500
