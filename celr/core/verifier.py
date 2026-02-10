import logging
from typing import Any, Optional
from celr.core.types import Step, StepType, TaskContext
from celr.core.tools import ToolRegistry
from celr.core.llm import BaseLLMProvider

logger = logging.getLogger(__name__)

class Verifier:
    def __init__(self, tool_registry: ToolRegistry, llm: BaseLLMProvider):
        self.registry = tool_registry
        self.llm = llm

    def verify(self, step: Step, context: TaskContext) -> bool:
        """
        Verifies the output of a step.
        Returns True if successful, False otherwise.
        """
        # 1. Self-Verification (Execution)
        # If the step was code execution, did it run without error?
        if step.step_type == StepType.EXECUTION:
            # Simple heuristic: if output contains "Error", fail.
            if "Error" in (step.output or "") or "Exception" in (step.output or ""):
                step.verification_notes = "Execution Output contained error keywords."
                return False
            return True

        # 2. LLM-Based Verification (Reasoning)
        # Ask a separate (or same) LLM to critique the result.
        # "Peer Review" Pattern
        
        prompt = f"""
        VERIFICATION TASK
        Original Goal: {step.description}
        Generated Output: {step.output}
        
        Does the output satisfy the goal? 
        Reply with YES or NO. If NO, explain why.
        """
        
        # In production, use a cheap model or a specialized verifier prompt
        critique, usage = self.llm.generate(prompt)
        
        if "YES" in critique.upper():
            step.verification_notes = critique
            return True
        else:
            step.verification_notes = f"Verification Failed: {critique}"
            return False
