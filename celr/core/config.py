"""
CELR Configuration System.

Uses Pydantic BaseSettings to load configuration from:
  1. Environment variables (CELR_ prefix)
  2. .env file
  3. Defaults

Usage:
    config = CELRConfig()  # auto-loads from env
    config = CELRConfig(budget_limit=1.0, verbose=True)  # explicit
"""

import logging
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from celr.core.types import ModelConfig

logger = logging.getLogger(__name__)


class CELRConfig(BaseSettings):
    """Master configuration for CELR."""

    model_config = SettingsConfigDict(
        env_prefix="CELR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Budget
    budget_limit: float = Field(default=0.50, description="Max USD to spend")
    max_retries: int = Field(default=3, description="Max retry attempts per step")

    # Models
    small_model: str = Field(default="ollama/llama3", description="Default small/free model")
    large_model: str = Field(default="gpt-4o", description="Default large/expensive model")
    mid_model: str = Field(default="gpt-4o-mini", description="Mid-tier model")

    # Escalation thresholds
    difficulty_threshold_high: float = Field(default=0.7, description="Difficulty ≥ this → expensive model")
    difficulty_threshold_mid: float = Field(default=0.4, description="Difficulty ≥ this → mid model")
    min_budget_for_escalation: float = Field(default=0.10, description="Reserve for escalation")

    # Logging & output
    log_dir: str = Field(default=".celr_logs", description="Directory for trajectory logs")
    verbose: bool = Field(default=False, description="Enable verbose/debug logging")

    # Safety
    max_output_bytes: int = Field(default=10240, description="Max tool output size")
    exec_timeout_seconds: int = Field(default=10, description="Tool execution timeout")

    # Optimization (Ollama)
    ollama_num_ctx: int = Field(default=4096, description="Ollama context window (lower to save VRAM)")
    ollama_keep_alive: str = Field(default="5m", description="How long to keep model loaded (e.g. 5m, 0)")

    def get_model_tiers(self) -> List[ModelConfig]:
        """Build the model tier list from config values."""
        return [
            ModelConfig(
                name=self.small_model,
                provider=self._infer_provider(self.small_model),
                cost_per_million_input_tokens=0.0,
                cost_per_million_output_tokens=0.0,
                ollama_num_ctx=self.ollama_num_ctx,
                ollama_keep_alive=self.ollama_keep_alive,
            ),
            ModelConfig(
                name=self.mid_model,
                provider=self._infer_provider(self.mid_model),
                cost_per_million_input_tokens=0.15,
                cost_per_million_output_tokens=0.60,
            ),
            ModelConfig(
                name=self.large_model,
                provider=self._infer_provider(self.large_model),
                cost_per_million_input_tokens=5.0,
                cost_per_million_output_tokens=15.0,
                supports_tools=True,
            ),
        ]

    @staticmethod
    def _infer_provider(model_name: str) -> str:
        """Infer provider from model name prefix."""
        if model_name.startswith("ollama/"):
            return "ollama"
        elif model_name.startswith("claude") or model_name.startswith("anthropic/"):
            return "anthropic"
        elif model_name.startswith("gpt") or model_name.startswith("o1") or model_name.startswith("o3"):
            return "openai"
        elif model_name.startswith("gemini/"):
            return "google"
        elif model_name.startswith("groq/"):
            return "groq"
        elif model_name.startswith("deepseek/"):
            return "deepseek"
        else:
            return "openai"  # Default fallback

    def setup_logging(self) -> None:
        """Configure Python logging based on verbose flag."""
        level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        # Quiet noisy libraries
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("litellm").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logger.debug(f"Logging configured: level={logging.getLevelName(level)}")
