"""Tests for celr.core.types — Step, Plan, TaskContext, ModelConfig."""

import pytest
from celr.core.types import (
    TaskStatus, StepType, EscalationTier,
    Step, Plan, TaskContext, ModelConfig,
)


class TestStep:
    def test_default_values(self):
        step = Step(description="Test")
        assert step.status == TaskStatus.PENDING
        assert step.step_type == StepType.REASONING
        assert step.estimated_difficulty == 0.5
        assert step.retry_count == 0
        assert step.max_retries == 3
        assert step.escalation_tier is None

    def test_custom_values(self):
        step = Step(
            id="custom-id",
            description="Custom step",
            step_type=StepType.EXECUTION,
            estimated_difficulty=0.9,
            max_retries=5,
        )
        assert step.id == "custom-id"
        assert step.step_type == StepType.EXECUTION
        assert step.max_retries == 5


class TestPlan:
    def test_get_runnable_steps_no_deps(self):
        s1 = Step(id="s1", description="Step 1")
        s2 = Step(id="s2", description="Step 2")
        plan = Plan(items=[s1, s2], original_goal="Test")
        
        runnable = plan.get_runnable_steps()
        assert len(runnable) == 2

    def test_get_runnable_steps_with_deps(self):
        s1 = Step(id="s1", description="Step 1")
        s2 = Step(id="s2", description="Step 2", dependencies=["s1"])
        plan = Plan(items=[s1, s2], original_goal="Test")
        
        runnable = plan.get_runnable_steps()
        assert len(runnable) == 1
        assert runnable[0].id == "s1"

    def test_get_runnable_after_completion(self):
        s1 = Step(id="s1", description="Step 1", status=TaskStatus.COMPLETED)
        s2 = Step(id="s2", description="Step 2", dependencies=["s1"])
        plan = Plan(items=[s1, s2], original_goal="Test")
        
        runnable = plan.get_runnable_steps()
        assert len(runnable) == 1
        assert runnable[0].id == "s2"


class TestTaskContext:
    def test_budget_remaining(self, sample_context):
        assert sample_context.budget_remaining == 1.00
        sample_context.current_spread_usd = 0.30
        assert sample_context.budget_remaining == 0.70

    def test_log(self, sample_context):
        sample_context.log("Test message")
        assert len(sample_context.execution_history) == 1
        assert "Test message" in sample_context.execution_history[0]


class TestEscalationTier:
    def test_values(self):
        assert EscalationTier.LOCAL.value == "LOCAL"
        assert EscalationTier.CHEAP_REMOTE.value == "CHEAP_REMOTE"
        assert EscalationTier.EXPENSIVE_REMOTE.value == "EXPENSIVE_REMOTE"


class TestModelConfig:
    def test_defaults(self):
        config = ModelConfig(name="test", provider="test")
        assert config.context_window == 4096
        assert config.supports_tools is False
        assert config.is_reasoning_model is False
