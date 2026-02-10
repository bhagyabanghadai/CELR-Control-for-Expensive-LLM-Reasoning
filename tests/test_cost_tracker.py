"""Tests for celr.core.cost_tracker — Budget enforcement."""

import pytest
from celr.core.cost_tracker import CostTracker
from celr.core.types import TaskContext


class TestCostTracker:
    def test_add_cost(self, sample_context, cost_tracker):
        cost_tracker.add_cost(0.10)
        assert sample_context.current_spread_usd == 0.10

    def test_check_remaining(self, sample_context, cost_tracker):
        cost_tracker.add_cost(0.30)
        assert cost_tracker.check_remaining_budget() == 0.70

    def test_can_afford_yes(self, cost_tracker):
        assert cost_tracker.can_afford(0.50) is True

    def test_can_afford_no(self, sample_context, cost_tracker):
        cost_tracker.add_cost(0.90)
        assert cost_tracker.can_afford(0.20) is False

    def test_budget_exceeded_warning(self, sample_context, cost_tracker):
        cost_tracker.add_cost(1.50)  # Over budget
        # Should log a warning but not raise
        assert sample_context.current_spread_usd == 1.50
        assert any("WARNING" in msg for msg in sample_context.execution_history)

    def test_zero_budget(self):
        ctx = TaskContext(original_request="test", budget_limit_usd=0.0)
        tracker = CostTracker(ctx)
        assert tracker.check_remaining_budget() == 0.0
        assert tracker.can_afford(0.01) is False
