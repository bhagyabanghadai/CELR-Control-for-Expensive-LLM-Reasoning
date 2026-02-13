"""
Adaptive Cortex Router.

Decides whether a user query requires the full reasoning engine ("System 2")
or can be answered immediately by the LLM ("System 1").

This is the key to balancing "Best Results" with "Quick Results".
"""

from typing import Tuple, Dict, Any
from celr.core.llm import BaseLLMProvider
import json

class Router:
    def __init__(self, llm: BaseLLMProvider):
        self.llm = llm

    def classify(self, prompt: str) -> Tuple[str, str]:
        """
        Determine if the task is 'simple' or 'complex'.
        
        Returns:
            (category, reasoning)
            category: "DIRECT" or "REASONING"
        """
        router_prompt = (
            f"Analyze the following user query: \"{prompt}\"\n\n"
            "Is this a complex reasoning task that requires a multi-step plan, math, or coding?\n"
            "Or is it a simple greeting, factoid, or conversation?\n\n"
            "Respond with JSON only: {{ \"type\": \"DIRECT\" or \"REASONING\", \"reason\": \"...\" }}"
        )

        try:
            # Use a low temperature for classification
            # Note: We use the same LLM, but for local models this check is fast enough.
            response, _ = self.llm.generate(
                prompt=router_prompt,
                system_prompt="You are a classifier. Output JSON only."
            )
            
            # Naive parsing (improve with robust JSON parser if needed)
            cleaned = response.strip().replace("```json", "").replace("```", "")
            data = json.loads(cleaned)
            return data.get("type", "DIRECT"), data.get("reason", "")
            
        except Exception as e:
            # Default to reasoning if unsure (safe fallback), or direct for speed?
            # User wants speed. Default to DIRECT if classification fails/timeout.
            return "DIRECT", f"Classification failed: {e}"
