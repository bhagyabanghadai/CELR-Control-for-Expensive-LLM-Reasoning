"""
Escalation Manager — cost-aware model routing.

Decides which LLM tier handles each step based on:
  - Step difficulty (0.0 to 1.0)
  - Remaining budget
  - History (did a cheaper model already fail?)

Key improvements (Overhaul Phase O-3):
  - Configurable model tiers instead of hardcoded configs
  - get_provider() returns an actual LiteLLMProvider instance
  - Sorted by cost (cheapest first) for budget efficiency
"""

import logging
from typing import List, Optional

from celr.core.cost_tracker import CostTracker
from celr.core.exceptions import EscalationError
from celr.core.llm import BaseLLMProvider, LiteLLMProvider
from celr.core.types import EscalationTier, ModelConfig, Step

logger = logging.getLogger(__name__)

# Default model configurations (can be overridden via config)
DEFAULT_MODEL_TIERS: List[ModelConfig] = [
    ModelConfig(
        name="ollama/llama3",
        provider="ollama",
        cost_per_million_input_tokens=0.0,
        cost_per_million_output_tokens=0.0,
        context_window=8192,
    ),
    ModelConfig(
        name="gpt-4o-mini",
        provider="openai",
        cost_per_million_input_tokens=0.15,
        cost_per_million_output_tokens=0.60,
        context_window=128_000,
    ),
    ModelConfig(
        name="gpt-4o",
        provider="openai",
        cost_per_million_input_tokens=5.0,
        cost_per_million_output_tokens=15.0,
        context_window=128_000,
        supports_tools=True,
    ),
]


class EscalationManager:
    """
    Routes steps to the right model based on difficulty, budget, and history.
    
    Tiers:
      LOCAL            → free/cheap local model (Ollama)
      CHEAP_REMOTE     → budget cloud model (GPT-4o-mini)
      EXPENSIVE_REMOTE → full power cloud model (GPT-4o)
    """

    def __init__(
        self,
        cost_tracker: CostTracker,
        model_tiers: Optional[List[ModelConfig]] = None,
    ):
        self.tracker = cost_tracker
        # Sort by cost (cheapest first)
        self.tiers = sorted(
            model_tiers or DEFAULT_MODEL_TIERS,
            key=lambda m: m.cost_per_million_input_tokens + m.cost_per_million_output_tokens,
        )

        # Assign tier labels
        self._tier_map = {}
        for i, model in enumerate(self.tiers):
            if i == 0:
                self._tier_map[model.name] = EscalationTier.LOCAL
            elif i == len(self.tiers) - 1:
                self._tier_map[model.name] = EscalationTier.EXPENSIVE_REMOTE
            else:
                self._tier_map[model.name] = EscalationTier.CHEAP_REMOTE

        # Heuristics
        self.difficulty_threshold_high = 0.7  # → EXPENSIVE
        self.difficulty_threshold_mid = 0.4   # → CHEAP_REMOTE
        self.min_budget_for_escalation = 0.10

        logger.info(f"EscalationManager initialized with {len(self.tiers)} model tiers")

    def select_model(self, step: Step, force_expensive: bool = False) -> str:
        """
        Returns the name of the model to use for this step.
        
        Decision logic:
          1. Budget safety check
          2. Forced Escalation (Cortex Override)
          3. Budget drain loophole mitigation
          4. Difficulty-based routing
        """
        remaining = self.tracker.check_remaining_budget()

        # 1. Budget safety — if broke, force cheapest
        if remaining <= 0:
            logger.warning("Budget exhausted, forcing cheapest model")
            return self.tiers[0].name

        # 2. Forced Escalation (Adaptive Cortex Decision)
        if force_expensive:
            if self.tracker.can_afford(self.min_budget_for_escalation):
                logger.info("Adaptive Cortex forced escalation -> Expensive Model")
                return self.tiers[-1].name
            else:
                logger.warning("Cortex requested escalation but budget too low. Using cheapest.")
                return self.tiers[0].name

        # 3. Budget drain mitigation — save money for hard steps
        if remaining < 2 * self.min_budget_for_escalation:
            if step.estimated_difficulty > self.difficulty_threshold_mid:
                logger.info(
                    f"Budget low (${remaining:.4f}), saving for hard step → {self.tiers[-1].name}"
                )
                return self.tiers[-1].name

        # 4. Standard difficulty-based routing
        if step.estimated_difficulty >= self.difficulty_threshold_high:
            if self.tracker.can_afford(self.min_budget_for_escalation):
                return self.tiers[-1].name  # Expensive
            else:
                logger.warning(
                    f"Step needs expensive model but can't afford. Using cheapest."
                )
                return self.tiers[0].name

        if step.estimated_difficulty >= self.difficulty_threshold_mid and len(self.tiers) > 1:
            return self.tiers[1].name  # Mid-tier if available

        return self.tiers[0].name  # Default to cheapest

    def get_tier(self, model_name: str) -> EscalationTier:
        """Get the tier label for a model name."""
        return self._tier_map.get(model_name, EscalationTier.LOCAL)

    def get_provider(self, step: Step, force_expensive: bool = False) -> BaseLLMProvider:
        """
        Select model AND return an actual LiteLLMProvider instance.
        This is the main method the executor should call.
        """
        model_name = self.select_model(step, force_expensive=force_expensive)
        tier = self.get_tier(model_name)
        step.assigned_agent = model_name
        step.escalation_tier = tier.value

        # Find the config for this model
        config = next((m for m in self.tiers if m.name == model_name), None)
        if not config:
            raise EscalationError(f"No config found for model: {model_name}")

        logger.info(
            f"Routing step '{step.description[:50]}' "
            f"(difficulty={step.estimated_difficulty}) → {model_name} ({tier.value})"
        )

        return LiteLLMProvider(config)

    def get_config_for(self, model_name: str) -> Optional[ModelConfig]:
        """Get the ModelConfig for a given model name."""
        return next((m for m in self.tiers if m.name == model_name), None)
