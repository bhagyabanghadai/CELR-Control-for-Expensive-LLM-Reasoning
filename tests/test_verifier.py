"""Tests for celr.core.verifier — Output verification logic."""

import pytest
from celr.core.types import Step, StepType, TaskContext
from celr.core.verifier import Verifier
from celr.core.tools import ToolRegistry


class TestVerifier:
    def test_execution_step_passes_clean_output(self, mock_llm, tool_registry, sample_context):
        """Execution step with clean output should pass."""
        v = Verifier(tool_registry=tool_registry, llm=mock_llm)
        step = Step(id="s1", description="Run code", step_type=StepType.EXECUTION, output="42")
        assert v.verify(step, sample_context) is True

    def test_execution_step_fails_on_error(self, mock_llm, tool_registry, sample_context):
        """Execution step with Python error in output should fail."""
        v = Verifier(tool_registry=tool_registry, llm=mock_llm)
        step = Step(id="s1", description="Run code", step_type=StepType.EXECUTION, output="ZeroDivisionError: division by zero")
        assert v.verify(step, sample_context) is False

    def test_execution_step_fails_on_exception(self, mock_llm, tool_registry, sample_context):
        """Execution step with traceback in output should fail."""
        v = Verifier(tool_registry=tool_registry, llm=mock_llm)
        step = Step(id="s1", description="Run code", step_type=StepType.EXECUTION, output="Traceback (most recent call last):\n  File 'test.py', line 1\nTypeError: int")
        assert v.verify(step, sample_context) is False

    def test_reasoning_step_llm_says_yes(self, mock_llm_yes, tool_registry, sample_context):
        """Reasoning step should pass when LLM says YES."""
        v = Verifier(tool_registry=tool_registry, llm=mock_llm_yes)
        step = Step(id="s1", description="Explain gravity", step_type=StepType.REASONING, output="Gravity is...")
        assert v.verify(step, sample_context) is True

    def test_reasoning_step_llm_says_no(self, mock_llm_no, tool_registry, sample_context):
        """Reasoning step should fail when LLM says NO."""
        v = Verifier(tool_registry=tool_registry, llm=mock_llm_no)
        step = Step(id="s1", description="Explain gravity", step_type=StepType.REASONING, output="Wrong answer")
        assert v.verify(step, sample_context) is False
        assert "Verification Failed" in step.verification_notes


class TestVerifierEdgeCases:
    def test_empty_output_fails(self, mock_llm, tool_registry, sample_context):
        """Empty output should fail verification (improved sanity check)."""
        v = Verifier(tool_registry=tool_registry, llm=mock_llm)
        step = Step(id="s1", description="Run code", step_type=StepType.EXECUTION, output="")
        assert v.verify(step, sample_context) is False
        assert "empty" in step.verification_notes.lower()

    def test_none_output_fails(self, mock_llm, tool_registry, sample_context):
        """None output should fail verification (improved sanity check)."""
        v = Verifier(tool_registry=tool_registry, llm=mock_llm)
        step = Step(id="s1", description="Run code", step_type=StepType.EXECUTION, output=None)
        assert v.verify(step, sample_context) is False
        assert "empty" in step.verification_notes.lower()

    def test_execution_step_passes_success_no_output(self, mock_llm, tool_registry, sample_context):
        """'Success (No Output)' from tool registry should pass."""
        v = Verifier(tool_registry=tool_registry, llm=mock_llm)
        step = Step(id="s1", description="Run code", step_type=StepType.EXECUTION, output="Success (No Output)")
        assert v.verify(step, sample_context) is True

    def test_confidence_extraction(self, tool_registry):
        """Verifier should extract confidence from LLM response."""
        from tests.conftest import MockLLMProvider
        llm = MockLLMProvider(response="VERDICT: YES\nCONFIDENCE: 0.95\nREASON: Looks correct.")
        v = Verifier(tool_registry=tool_registry, llm=llm)
        step = Step(id="s1", description="Test", step_type=StepType.REASONING, output="Some output")
        ctx = TaskContext(original_request="test", budget_limit_usd=1.0)
        assert v.verify(step, ctx) is True
        assert "0.95" in step.verification_notes

    def test_specific_python_errors_detected(self, mock_llm, tool_registry, sample_context):
        """Specific Python exception types should be caught."""
        v = Verifier(tool_registry=tool_registry, llm=mock_llm)
        error_types = [
            "NameError: name 'x' is not defined",
            "TypeError: unsupported operand",
            "ImportError: No module named 'foo'",
            "SyntaxError: invalid syntax",
        ]
        for error_output in error_types:
            step = Step(id="s1", description="Run code", step_type=StepType.EXECUTION, output=error_output)
            assert v.verify(step, sample_context) is False, f"Should fail on: {error_output}"
