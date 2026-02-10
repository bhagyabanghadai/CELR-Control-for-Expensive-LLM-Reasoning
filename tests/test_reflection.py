"""Tests for celr.core.reflection — Failure analysis and retry decisions."""

import pytest
from celr.core.reflection import SelfReflection
from celr.core.types import Step


class TestSelfReflection:
    def test_analyze_failure_calls_llm(self, mock_llm, sample_context):
        """analyze_failure should call the LLM and log the result."""
        r = SelfReflection(llm=mock_llm)
        step = Step(id="s1", description="Failed step", output="Error occurred")
        
        result = r.analyze_failure(step, sample_context)
        assert result == "Mock LLM response"
        assert mock_llm.call_count == 1

    def test_analyze_failure_includes_context(self, mock_llm, sample_context):
        """The prompt should include step description and error info."""
        r = SelfReflection(llm=mock_llm)
        step = Step(id="s1", description="Parse JSON data", verification_notes="Invalid format")
        
        r.analyze_failure(step, sample_context)
        assert "Parse JSON data" in mock_llm.last_prompt
        assert "Invalid format" in mock_llm.last_prompt


class TestShouldRetry:
    def test_retry_under_max(self, mock_llm):
        """Should retry when under max attempts."""
        r = SelfReflection(llm=mock_llm)
        step = Step(id="s1", description="Test")
        assert r.should_retry(step, attempt_count=0) is True
        assert r.should_retry(step, attempt_count=1) is True
        assert r.should_retry(step, attempt_count=2) is True

    def test_no_retry_at_max(self, mock_llm):
        """Should NOT retry at max attempts."""
        r = SelfReflection(llm=mock_llm)
        step = Step(id="s1", description="Test")
        assert r.should_retry(step, attempt_count=3) is False

    def test_no_retry_over_max(self, mock_llm):
        """Should NOT retry over max attempts."""
        r = SelfReflection(llm=mock_llm)
        step = Step(id="s1", description="Test")
        assert r.should_retry(step, attempt_count=10) is False
