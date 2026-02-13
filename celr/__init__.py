"""
CELR: Control for Expensive LLM Reasoning

A meta-brain for AI agents that optimizes cost and quality
by routing tasks to the right model at the right time.
"""

__version__ = "0.1.0"

from celr.core.config import CELRConfig
from celr.core.types import TaskContext, Plan, Step, StepType, TaskStatus
from celr.core.executor import TaskExecutor
from celr.core.llm import BaseLLMProvider, LiteLLMProvider, LLMUsage
from celr.core.cost_tracker import CostTracker
from celr.core.escalation import EscalationManager
from celr.core.planner import Planner
from celr.core.reasoning import ReasoningCore
from celr.core.tools import ToolRegistry
from celr.core.verifier import Verifier
from celr.core.reflection import SelfReflection
from celr.core.exceptions import CELRError, BudgetExhaustedError

__all__ = [
    "__version__",
    "CELRConfig",
    "TaskContext",
    "Plan",
    "Step",
    "StepType",
    "TaskStatus",
    "TaskExecutor",
    "BaseLLMProvider",
    "LiteLLMProvider",
    "LLMUsage",
    "CostTracker",
    "EscalationManager",
    "Planner",
    "ReasoningCore",
    "ToolRegistry",
    "Verifier",
    "SelfReflection",
    "CELRError",
    "BudgetExhaustedError",
]
