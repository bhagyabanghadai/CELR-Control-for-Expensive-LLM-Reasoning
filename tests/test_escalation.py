"""Tests for celr.core.escalation — Model routing logic."""

import pytest
from celr.core.escalation import EscalationManager
from celr.core.types import EscalationTier, Step


class TestEscalationManager:
    def test_easy_step_routes_to_cheapest(self, escalation_manager, sample_step):
        """Easy step (difficulty=0.3) should route to cheapest model."""
        model = escalation_manager.select_model(sample_step)
        assert model == "mock-small"

    def test_hard_step_routes_to_expensive(self, escalation_manager, hard_step):
        """Hard step (difficulty=0.9) should route to expensive model."""
        model = escalation_manager.select_model(hard_step)
        assert model == "mock-large"

    def test_mid_difficulty_routes_to_mid(self, escalation_manager):
        """Mid-difficulty step should route to mid-tier."""
        step = Step(id="mid", description="Medium task", estimated_difficulty=0.5)
        model = escalation_manager.select_model(step)
        assert model == "mock-mid"

    def test_zero_budget_forces_cheapest(self, sample_context, cost_tracker, model_configs):
        """When budget is 0, always use cheapest model regardless of difficulty."""
        sample_context.current_spread_usd = sample_context.budget_limit_usd  # Budget gone
        em = EscalationManager(cost_tracker=cost_tracker, model_tiers=model_configs)
        hard = Step(id="hard", description="Hard task", estimated_difficulty=0.95)
        model = em.select_model(hard)
        assert model == "mock-small"

    def test_get_tier(self, escalation_manager):
        """Tier labels should be assigned correctly."""
        assert escalation_manager.get_tier("mock-small") == EscalationTier.LOCAL
        assert escalation_manager.get_tier("mock-large") == EscalationTier.EXPENSIVE_REMOTE

    def test_get_provider_returns_provider(self, escalation_manager, sample_step):
        """get_provider should return an actual provider instance."""
        from unittest.mock import patch, MagicMock
        # We can't actually create a LiteLLMProvider with mock config in tests
        # but we can test the routing logic
        with patch("celr.core.escalation.LiteLLMProvider") as MockProvider:
            MockProvider.return_value = MagicMock()
            provider = escalation_manager.get_provider(sample_step)
            assert provider is not None
            assert sample_step.assigned_agent is not None
