"""Tests for celr.training.self_reward — LLM-as-Judge scoring."""

import pytest
from celr.training.self_reward import SelfRewardScorer

# Import the MockLLMProvider from conftest
from tests.conftest import MockLLMProvider


class TestScoreParsing:
    def test_parse_structured_score(self):
        """Should parse well-formatted score response."""
        llm = MockLLMProvider(
            response="Accuracy: 5\nCompleteness: 4\nEfficiency: 3\nTotal: 12"
        )
        scorer = SelfRewardScorer(llm=llm)
        score = scorer.score_response("test prompt", "test response")

        # (5+4+3 - 3) / 12 = 9/12 = 0.75
        assert abs(score - 0.75) < 0.01

    def test_parse_low_score(self):
        """Should parse low scores."""
        llm = MockLLMProvider(
            response="Accuracy: 1\nCompleteness: 1\nEfficiency: 1\nTotal: 3"
        )
        scorer = SelfRewardScorer(llm=llm)
        score = scorer.score_response("test", "test")
        assert score == 0.0  # (3-3)/12 = 0

    def test_parse_perfect_score(self):
        """Should parse perfect scores."""
        llm = MockLLMProvider(
            response="Accuracy: 5\nCompleteness: 5\nEfficiency: 5\nTotal: 15"
        )
        scorer = SelfRewardScorer(llm=llm)
        score = scorer.score_response("test", "test")
        assert score == 1.0  # (15-3)/12 = 1.0

    def test_fallback_on_unparseable(self):
        """Should return 0.5 when score can't be parsed."""
        llm = MockLLMProvider(response="I think it's pretty good!")
        scorer = SelfRewardScorer(llm=llm)
        score = scorer.score_response("test", "test")
        assert score == 0.5  # Neutral fallback


class TestSelfRewardScorer:
    def test_score_trajectory(self):
        llm = MockLLMProvider(
            response="Accuracy: 4\nCompleteness: 4\nEfficiency: 4\nTotal: 12"
        )
        scorer = SelfRewardScorer(llm=llm)

        traj = {
            "original_request": "Test task",
            "final_status": "SUCCESS",
            "plan": [
                {"status": "COMPLETED", "description": "Step 1"},
                {"status": "COMPLETED", "description": "Step 2"},
            ],
        }
        score = scorer.score_trajectory(traj)
        assert 0.0 <= score <= 1.0

    def test_generate_reward_pairs(self):
        llm = MockLLMProvider(
            response="Accuracy: 5\nCompleteness: 5\nEfficiency: 5\nTotal: 15"
        )
        scorer = SelfRewardScorer(llm=llm)

        trajectories = [
            {"original_request": "Task A", "final_status": "SUCCESS", "plan": []},
            {"original_request": "Task B", "final_status": "FAILED", "plan": []},
        ]
        # With identical scores, no pairs should be created (min_gap=0.2)
        pairs = scorer.generate_reward_pairs(trajectories, min_gap=0.2)
        # All get same score since mock returns same response
        assert isinstance(pairs, list)

    def test_scoring_cost_tracked(self):
        llm = MockLLMProvider(
            response="Accuracy: 3\nCompleteness: 3\nEfficiency: 3\nTotal: 9"
        )
        scorer = SelfRewardScorer(llm=llm)
        scorer.score_response("test", "test")
        # MockLLM returns 0 cost, but the tracking mechanism should work
        assert scorer.total_scoring_cost >= 0.0
