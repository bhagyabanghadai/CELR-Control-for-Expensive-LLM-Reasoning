import json
from typing import Optional, List, Dict
from celr.core.types import TaskContext, Plan, Step, StepType
from celr.core.llm import BaseLLMProvider
from celr.core.prompts import DECOMPOSITION_SYSTEM_PROMPT, DIFFICULTY_ESTIMATION_PROMPT

class ReasoningCore:
    def __init__(self, llm: BaseLLMProvider):
        self.llm = llm

    def decompose(self, context: TaskContext) -> Plan:
        """Decompose a high-level task into a Plan."""
        prompt = f"Original Goal: {context.original_request}\n\nContext: {context.execution_history[-5:]}"
        
        response = self.llm.generate(
            prompt=prompt,
            system_prompt=DECOMPOSITION_SYSTEM_PROMPT
        )
        
        try:
            # Clean up potential markdown code blocks
            clean_response = response.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_response)
            
            # Convert JSON items to Step objects
            steps = []
            for item in data.get("items", []):
                steps.append(Step(**item))
                
            plan = Plan(
                original_goal=data.get("original_goal", context.original_request),
                items=steps
            )
            return plan
        except Exception as e:
            # Fallback for parsing errors
            context.log(f"Error parsing plan: {e}")
            raise ValueError(f"Failed to generate valid plan from LLM response: {response}")

    def estimate_difficulty(self, step: Step) -> float:
        """Estimate the difficulty of a single step."""
        prompt = DIFFICULTY_ESTIMATION_PROMPT.format(task_description=step.description)
        response = self.llm.generate(prompt=prompt)
        
        try:
            clean = response.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean)
            return float(data.get("difficulty_score", 0.5))
        except:
            return 0.5 # Default fallback
