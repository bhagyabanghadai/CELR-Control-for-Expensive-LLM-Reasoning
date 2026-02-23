import logging
import json
from typing import Dict, Any, List, Optional
from celr.core.prompts import (
    BLUEPRINT_ARCHITECT_PROMPT,
    CODER_AGENT_PROMPT,
    MATHEMATICIAN_AGENT_PROMPT,
    RESEARCHER_AGENT_PROMPT,
    CRITIC_AGENT_PROMPT
)

logger = logging.getLogger(__name__)

class SpecialistAgent:
    """Base class for specialized agents."""
    def __init__(self, role: str, model_name: str = "ollama/llama3.2"):
        self.role = role
        self.model_name = model_name
        self.system_prompt = self._get_system_prompt()

    def _get_system_prompt(self) -> str:
        """Returns the specific prompt for this agent role."""
        raise NotImplementedError 

    def format_prompt(self, task: str, context: Dict[str, Any]) -> str:
        """Formats the user prompt with context."""
        context_str = json.dumps(context, indent=2)
        return f"Task: {task}\n\nContext:\n{context_str}\n\nRole: {self.role}"

class BlueprintArchitect(SpecialistAgent):
    """Decomposes problems into a structured Blueprint schema."""
    def _get_system_prompt(self) -> str:
        return BLUEPRINT_ARCHITECT_PROMPT

class Coder(SpecialistAgent):
    """Python Expert. Writes efficient, correct code using PersistentRuntime."""
    def _get_system_prompt(self) -> str:
        return CODER_AGENT_PROMPT

class Mathematician(SpecialistAgent):
    """Math Expert. Solves problems step-by-step with explicit logic."""
    def _get_system_prompt(self) -> str:
        return MATHEMATICIAN_AGENT_PROMPT

class Researcher(SpecialistAgent):
    """Knowledge Expert. Provides accurate, fact-based answers."""
    def _get_system_prompt(self) -> str:
        return RESEARCHER_AGENT_PROMPT

class Critic(SpecialistAgent):
    """Verifies steps and critiques logic before execution proceeds."""
    def _get_system_prompt(self) -> str:
        return CRITIC_AGENT_PROMPT

class AgentFactory:
    """Factory to create agents based on role."""
    @staticmethod
    def get_agent(role: str) -> SpecialistAgent:
        if role == "ARCHITECT":
            return BlueprintArchitect(role)
        elif role == "CODER":
            return Coder(role)
        elif role == "MATHEMATICIAN":
            return Mathematician(role)
        elif role == "RESEARCHER":
            return Researcher(role)
        elif role == "CRITIC":
            return Critic(role)
        else:
            raise ValueError(f"Unknown agent role: {role}")
