"""
CELR Core - Control for Expensive LLM Reasoning

Re-exports all public interfaces for clean imports:
    from celr.core import TaskContext, Plan, Step, CostTracker, TaskExecutor
"""

# Data structures
from celr.core.types import (
    TaskStatus,
    StepType,
    EscalationTier,
    Step,
    Plan,
    TaskContext,
    ModelConfig,
)

# Exceptions
from celr.core.exceptions import (
    CELRError,
    BudgetExhaustedError,
    PlanningError,
    EscalationError,
    ToolExecutionError,
    LLMProviderError,
    VerificationError,
)

# Core components
from celr.core.cost_tracker import CostTracker
from celr.core.llm import BaseLLMProvider, LiteLLMProvider
from celr.core.reasoning import ReasoningCore
from celr.core.planner import Planner
from celr.core.executor import TaskExecutor
from celr.core.escalation import EscalationManager
from celr.core.tools import ToolRegistry
from celr.core.verifier import Verifier
from celr.core.reflection import SelfReflection
from celr.core.logger import TrajectoryLogger
from celr.core.trainer import Trainer

__all__ = [
    # Types
    "TaskStatus", "StepType", "EscalationTier", "Step", "Plan", "TaskContext", "ModelConfig",
    # Exceptions
    "CELRError", "BudgetExhaustedError", "PlanningError", "EscalationError",
    "ToolExecutionError", "LLMProviderError", "VerificationError",
    # Components
    "CostTracker", "BaseLLMProvider", "LiteLLMProvider", "ReasoningCore",
    "Planner", "TaskExecutor", "EscalationManager", "ToolRegistry",
    "Verifier", "SelfReflection", "TrajectoryLogger", "Trainer",
]
