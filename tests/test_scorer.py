"""Tests for celr.training.scorer — Trajectory quality scoring."""

import pytest
from celr.training.scorer import TrajectoryScorer, TrajectoryScore


def _make_trajectory(status="SUCCESS", retries=0, cost=0.10, budget=1.0, steps=3, completed=3):
    """Helper to create a trajectory dict for testing."""
    plan = []
    for i in range(steps):
        plan.append({
            "id": f"step-{i}",
            "status": "COMPLETED" if i < completed else "FAILED",
            "retry_count": retries // max(steps, 1),
        })
    return {
        "task_id": f"test-{status.lower()}",
        "original_request": "Test task",
        "final_status": status,
        "plan": plan,
        "budget_limit_usd": budget,
        "total_cost_usd": cost,
    }


class TestTrajectoryScorer:
    def test_perfect_trajectory(self):
        scorer = TrajectoryScorer()
        traj = _make_trajectory(status="SUCCESS", retries=0, cost=0.0)
        score = scorer.score(traj)

        assert score.success_score == 1.0
        assert score.efficiency_score == 1.0
        assert score.cost_score == 1.0
        assert score.step_completion_rate == 1.0
        assert score.composite_score == 1.0

    def test_failed_trajectory(self):
        scorer = TrajectoryScorer()
        traj = _make_trajectory(status="FAILED", retries=5, cost=0.80, completed=1)
        score = scorer.score(traj)

        assert score.success_score == 0.0
        assert score.efficiency_score < 1.0
        assert score.step_completion_rate < 1.0
        assert score.composite_score < 0.5

    def test_retries_reduce_efficiency(self):
        scorer = TrajectoryScorer()
        no_retry = scorer.score(_make_trajectory(retries=0))
        many_retries = scorer.score(_make_trajectory(retries=10))

        assert no_retry.efficiency_score > many_retries.efficiency_score

    def test_cost_affects_score(self):
        scorer = TrajectoryScorer()
        cheap = scorer.score(_make_trajectory(cost=0.01, budget=1.0))
        expensive = scorer.score(_make_trajectory(cost=0.99, budget=1.0))

        assert cheap.cost_score > expensive.cost_score

    def test_zero_budget(self):
        scorer = TrajectoryScorer()
        score = scorer.score(_make_trajectory(budget=0.0, cost=0.0))
        assert score.cost_score == 1.0

    def test_rank_orders_correctly(self):
        scorer = TrajectoryScorer()
        trajectories = [
            _make_trajectory(status="FAILED", retries=5, cost=0.9, completed=0),
            _make_trajectory(status="SUCCESS", retries=0, cost=0.01),
            _make_trajectory(status="SUCCESS", retries=3, cost=0.5),
        ]
        ranked = scorer.rank(trajectories)

        assert ranked[0].composite_score >= ranked[1].composite_score
        assert ranked[1].composite_score >= ranked[2].composite_score

    def test_filter_high_quality(self):
        scorer = TrajectoryScorer()
        trajectories = [
            _make_trajectory(status="SUCCESS", retries=0, cost=0.01),
            _make_trajectory(status="FAILED", retries=5, cost=0.9, completed=0),
        ]
        high_q = scorer.filter_high_quality(trajectories, threshold=0.5)
        assert len(high_q) >= 1
        assert all(s.composite_score >= 0.5 for s in high_q)


class TestTrajectoryScore:
    def test_dataclass_fields(self):
        score = TrajectoryScore(
            trajectory_id="t1",
            original_request="test",
            final_status="SUCCESS",
        )
        assert score.composite_score == 0.0
        assert score.total_retries == 0
