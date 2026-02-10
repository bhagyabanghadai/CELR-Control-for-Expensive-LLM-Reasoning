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
        """Execution step with 'Error' in output should fail."""
        v = Verifier(tool_registry=tool_registry, llm=mock_llm)
        step = Step(id="s1", description="Run code", step_type=StepType.EXECUTION, output="Error: division by zero")
        assert v.verify(step, sample_context) is False

    def test_execution_step_fails_on_exception(self, mock_llm, tool_registry, sample_context):
        """Execution step with 'Exception' in output should fail."""
        v = Verifier(tool_registry=tool_registry, llm=mock_llm)
        step = Step(id="s1", description="Run code", step_type=StepType.EXECUTION, output="Exception occurred")
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
    def test_empty_output_execution(self, mock_llm, tool_registry, sample_context):
        """Empty output for execution step should pass (no error keywords)."""
        v = Verifier(tool_registry=tool_registry, llm=mock_llm)
        step = Step(id="s1", description="Run code", step_type=StepType.EXECUTION, output="")
        assert v.verify(step, sample_context) is True

    def test_none_output_execution(self, mock_llm, tool_registry, sample_context):
        """None output should be handled safely."""
        v = Verifier(tool_registry=tool_registry, llm=mock_llm)
        step = Step(id="s1", description="Run code", step_type=StepType.EXECUTION, output=None)
        assert v.verify(step, sample_context) is True
