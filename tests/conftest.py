"""
Shared pytest fixtures for CELR test suite.

Provides mock LLM, sample contexts, plans, steps, and tool registry
so individual test files can focus on behavior, not setup.
"""

import pytest
from unittest.mock import MagicMock, patch

from celr.core.types import TaskContext, Plan, Step, StepType, TaskStatus, ModelConfig
from celr.core.llm import BaseLLMProvider, LLMUsage, LiteLLMProvider
from celr.core.cost_tracker import CostTracker
from celr.core.escalation import EscalationManager
from celr.core.tools import ToolRegistry
from celr.core.verifier import Verifier
from celr.core.reflection import SelfReflection


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM that returns configurable responses without network calls."""

    def __init__(self, response: str = "Mock LLM response", tokens: int = 100):
        self._response = response
        self._tokens = tokens
        self.call_count = 0
        self.last_prompt = None

    def generate(self, prompt, system_prompt=None, tools=None):
        self.call_count += 1
        self.last_prompt = prompt
        usage = LLMUsage(
            prompt_tokens=self._tokens // 2,
            completion_tokens=self._tokens // 2,
            total_tokens=self._tokens,
        )
        return self._response, usage

    def calculate_cost(self, usage):
        # Free mock
        return 0.0


@pytest.fixture
def mock_llm():
    """A mock LLM that returns a default response."""
    return MockLLMProvider()


@pytest.fixture
def mock_llm_yes():
    """A mock LLM that returns YES (for verification tests)."""
    return MockLLMProvider(response="YES, the output is correct.")


@pytest.fixture
def mock_llm_no():
    """A mock LLM that returns NO (for verification failure tests)."""
    return MockLLMProvider(response="NO, the output does not match the goal.")


@pytest.fixture
def sample_context():
    """A fresh TaskContext for testing."""
    return TaskContext(
        original_request="Test task: compute 2+2",
        budget_limit_usd=1.00,
    )


@pytest.fixture
def low_budget_context():
    """A context with very little budget left."""
    ctx = TaskContext(
        original_request="Test task with low budget",
        budget_limit_usd=0.05,
    )
    ctx.current_spread_usd = 0.04  # Only $0.01 remaining
    return ctx


@pytest.fixture
def sample_step():
    """A basic REASONING step."""
    return Step(
        id="step-1",
        description="Compute the sum of 2+2",
        step_type=StepType.REASONING,
        estimated_difficulty=0.3,
    )


@pytest.fixture
def hard_step():
    """A high-difficulty REASONING step."""
    return Step(
        id="step-hard",
        description="Design a distributed database architecture",
        step_type=StepType.REASONING,
        estimated_difficulty=0.9,
    )


@pytest.fixture
def execution_step():
    """An EXECUTION (tool use) step."""
    return Step(
        id="step-exec",
        description="TOOL:python_repl:print(2+2)",
        step_type=StepType.EXECUTION,
        estimated_difficulty=0.2,
    )


@pytest.fixture
def sample_plan(sample_step):
    """A simple plan with one step."""
    return Plan(
        items=[sample_step],
        original_goal="Compute 2+2",
    )


@pytest.fixture
def multi_step_plan():
    """A plan with multiple dependent steps."""
    step1 = Step(id="s1", description="Research the topic", step_type=StepType.REASONING)
    step2 = Step(id="s2", description="Write code", step_type=StepType.EXECUTION, dependencies=["s1"])
    step3 = Step(id="s3", description="Verify results", step_type=StepType.VERIFICATION, dependencies=["s2"])
    return Plan(items=[step1, step2, step3], original_goal="Full workflow test")


@pytest.fixture
def cost_tracker(sample_context):
    """A CostTracker bound to sample_context."""
    return CostTracker(sample_context)


@pytest.fixture
def tool_registry():
    """A fresh ToolRegistry with default tools."""
    return ToolRegistry()


@pytest.fixture
def model_configs():
    """The default model tier configs for testing."""
    return [
        ModelConfig(name="mock-small", provider="mock", cost_per_million_input_tokens=0.0, cost_per_million_output_tokens=0.0),
        ModelConfig(name="mock-mid", provider="mock", cost_per_million_input_tokens=0.15, cost_per_million_output_tokens=0.60),
        ModelConfig(name="mock-large", provider="mock", cost_per_million_input_tokens=5.0, cost_per_million_output_tokens=15.0),
    ]


@pytest.fixture
def escalation_manager(cost_tracker, model_configs):
    """EscalationManager with mock model configs."""
    return EscalationManager(cost_tracker=cost_tracker, model_tiers=model_configs)
