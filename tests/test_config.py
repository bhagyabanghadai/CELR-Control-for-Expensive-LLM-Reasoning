"""Tests for celr.core.config — Configuration system."""

import pytest
from celr.core.config import CELRConfig


class TestCELRConfig:
    def test_defaults(self):
        config = CELRConfig()
        assert config.budget_limit == 0.50
        assert config.max_retries == 3
        assert config.small_model == "ollama/llama3"
        assert config.large_model == "gpt-4o"
        assert config.verbose is False

    def test_override(self):
        config = CELRConfig(budget_limit=2.0, verbose=True, small_model="ollama/phi3")
        assert config.budget_limit == 2.0
        assert config.verbose is True
        assert config.small_model == "ollama/phi3"

    def test_get_model_tiers(self):
        config = CELRConfig()
        tiers = config.get_model_tiers()
        assert len(tiers) == 3
        assert tiers[0].name == config.small_model
        assert tiers[2].name == config.large_model

    def test_infer_provider(self):
        assert CELRConfig._infer_provider("ollama/llama3") == "ollama"
        assert CELRConfig._infer_provider("gpt-4o") == "openai"
        assert CELRConfig._infer_provider("claude-3-sonnet") == "anthropic"
        assert CELRConfig._infer_provider("gemini/pro") == "google"
        assert CELRConfig._infer_provider("unknown-model") == "openai"  # fallback
