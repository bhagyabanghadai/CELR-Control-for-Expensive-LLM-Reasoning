import json
import logging
from typing import Optional, List, Dict
from celr.core.types import TaskContext, Plan, Step, StepType
from celr.core.llm import BaseLLMProvider
from celr.core.prompts import DECOMPOSITION_SYSTEM_PROMPT, DIFFICULTY_ESTIMATION_PROMPT
from celr.core.exceptions import PlanningError

logger = logging.getLogger(__name__)



class ReasoningCore:
    def __init__(self, llm: BaseLLMProvider):
        self.llm = llm

    def decompose(self, context: TaskContext) -> Plan:
        """Decompose a high-level task into a Plan."""
        prompt = f"Original Goal: {context.original_request}\n\nContext: {context.execution_history[-5:]}"
        
        response, usage = self.llm.generate(
            prompt=prompt,
            system_prompt=DECOMPOSITION_SYSTEM_PROMPT
        )
        
        try:
            # Robust extraction of JSON object/array
            import re
            json_match = re.search(r'(\{.*\}|\[.*\])', response, re.DOTALL)
            if json_match:
                clean_response = json_match.group(1)
            else:
                # Fallback to basic stripping if regex fails
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
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Log the actual error before raising
            logger.error(f"Failed to parse plan from LLM response: {e}")
            logger.debug(f"Raw LLM response: {response[:500]}")
            raise PlanningError(
                message=f"Failed to generate valid plan: {e}",
                raw_response=response,
            ) from e

    def estimate_difficulty(self, step: Step) -> float:
        """Estimate the difficulty of a single step."""
        prompt = DIFFICULTY_ESTIMATION_PROMPT.format(task_description=step.description)
        response, usage = self.llm.generate(prompt=prompt)
        
        try:
            import re
            json_match = re.search(r'(\{.*\}|\[.*\])', response, re.DOTALL)
            if json_match:
                clean = json_match.group(1)
            else:
                clean = response.replace("```json", "").replace("```", "").strip()
                
            data = json.loads(clean)
            return float(data.get("difficulty_score", 0.5))
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse difficulty score, using default 0.5: {e}")
            return 0.5  # Default fallback
