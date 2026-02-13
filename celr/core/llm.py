"""
LLM Provider abstraction layer.

Provides BaseLLMProvider (interface) and LiteLLMProvider (real implementation).
Uses LiteLLM for unified access to 100+ LLM providers (OpenAI, Anthropic, Ollama, etc.)

Key improvements (Overhaul Phase O-2):
- Returns (text, usage) tuple for exact cost tracking
- Retry with exponential backoff via tenacity
- Proper exception handling (no bare except)
- Uses response.usage for token counts instead of len()/4 estimation
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import litellm
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from celr.core.exceptions import LLMProviderError
from celr.core.types import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class LLMUsage:
    """Token usage from an LLM call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class BaseLLMProvider:
    """Abstract base class for LLM providers."""

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
    ) -> Tuple[str, LLMUsage]:
        """
        Generate a completion.
        
        Returns:
            Tuple of (response_text, usage)
        """
        raise NotImplementedError

    def calculate_cost(self, usage: LLMUsage) -> float:
        """Calculate cost from actual token usage."""
        raise NotImplementedError


class LiteLLMProvider(BaseLLMProvider):
    """
    Real LLM provider using LiteLLM.
    Supports OpenAI, Anthropic, Ollama, and 100+ other backends.
    """

    def __init__(self, config: ModelConfig):
        self.config = config

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((
            litellm.exceptions.RateLimitError,
            litellm.exceptions.APIConnectionError,
            litellm.exceptions.ServiceUnavailableError,
        )),
        before_sleep=lambda retry_state: logging.getLogger(__name__).warning(
            f"LLM call failed (attempt {retry_state.attempt_number}), retrying..."
        ),
    )
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
    ) -> Tuple[str, LLMUsage]:
        """
        Generate a completion with retry logic.
        
        Returns:
            Tuple of (response_text, LLMUsage with actual token counts)
        
        Raises:
            LLMProviderError: If all retries fail
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = litellm.completion(
                model=self.config.name,
                messages=messages,
            )

            text = response.choices[0].message.content or ""
            
            # Extract REAL token usage from the API response
            usage = LLMUsage(
                prompt_tokens=getattr(response.usage, "prompt_tokens", 0),
                completion_tokens=getattr(response.usage, "completion_tokens", 0),
                total_tokens=getattr(response.usage, "total_tokens", 0),
            )

            logger.debug(
                f"LLM call: model={self.config.name}, "
                f"tokens={usage.total_tokens}, "
                f"prompt={usage.prompt_tokens}, completion={usage.completion_tokens}"
            )

            return text, usage

        except (litellm.exceptions.RateLimitError,
                litellm.exceptions.APIConnectionError,
                litellm.exceptions.ServiceUnavailableError):
            # Let tenacity handle these (re-raise for retry)
            raise
        except Exception as e:
            logger.error(f"LLM call failed: {self.config.name}: {e}")
            raise LLMProviderError(
                message=f"LLM generation failed: {e}",
                provider=self.config.provider,
                model=self.config.name,
            ) from e

    def calculate_cost(self, usage: LLMUsage) -> float:
        """
        Calculate cost from REAL token usage (not estimates).
        
        Uses the actual prompt_tokens and completion_tokens from the API response.
        """
        if self.config.provider == "ollama" or "local" in self.config.name:
            return 0.0

        try:
            cost = litellm.completion_cost(
                model=self.config.name,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
            )
            return cost
        except (ValueError, KeyError, TypeError) as e:
            # Fallback: manual calculation from config rates
            p_cost = (usage.prompt_tokens / 1_000_000) * self.config.cost_per_million_input_tokens
            c_cost = (usage.completion_tokens / 1_000_000) * self.config.cost_per_million_output_tokens
            return p_cost + c_cost
