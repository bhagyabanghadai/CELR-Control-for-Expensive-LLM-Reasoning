from typing import Any, List, Optional, Dict
from abc import ABC, abstractmethod
import os
import litellm
from celr.core.types import ModelConfig

class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, tools: Optional[List[Dict]] = None) -> str:
        """Synchronous generation."""
        pass
        
    @abstractmethod
    def calculate_cost(self, prompt: str, completion_text: str) -> float:
        """Estimate cost of the call."""
        pass

class LiteLLMProvider(BaseLLMProvider):
    def __init__(self, config: ModelConfig):
        self.config = config
        
    def generate(self, prompt: str, system_prompt: Optional[str] = None, tools: Optional[List[Dict]] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Call LiteLLM
        response = litellm.completion(
            model=self.config.name,
            messages=messages,
            # tools=tools if self.config.supports_tools else None
        )
        return response.choices[0].message.content

    def calculate_cost(self, prompt: str, completion_text: str) -> float:
        try:
            # simple fallback estimation if we don't have exact usage from last call
            # In production, we'd return (response, usage) from generate()
            prompt_tokens = len(prompt) / 4
            completion_tokens = len(completion_text) / 4
            
            # LiteLLM helper if available, or manual math
            cost = litellm.completion_cost(
                model=self.config.name, 
                completion_response=None, # Cannot calc exact without response obj
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens
            )
            return cost
        except:
            # Fallback manual calculation based on config
            p_cost = (len(prompt)/4) / 1_000_000 * self.config.cost_per_million_input_tokens
            c_cost = (len(completion_text)/4) / 1_000_000 * self.config.cost_per_million_output_tokens
            return p_cost + c_cost
