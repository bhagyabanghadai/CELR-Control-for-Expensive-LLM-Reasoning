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

    def _clean_json(self, text: str) -> str:
        """Extract and clean JSON from LLM output, handling common formatting issues."""
        import re
        # Strip markdown code fences
        text = text.replace("```json", "").replace("```", "").strip()
        # Find the outermost JSON object or array
        json_match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)
        # Remove trailing commas before } or ]
        text = re.sub(r',\s*([}\]])', r'\1', text)
        # Remove single-line comments (// ...)
        text = re.sub(r'//[^\n]*', '', text)
        return text

    def decompose(self, context: TaskContext) -> Plan:
        """Decompose a high-level task into a Plan with retry on JSON errors."""
        max_plan_retries = 3
        last_error = None

        for plan_attempt in range(max_plan_retries):
            prompt = f"Original Goal: {context.original_request}\n\nContext: {context.execution_history[-5:]}"

            if last_error and plan_attempt > 0:
                prompt += (
                    f"\n\nIMPORTANT: Your previous response had a JSON formatting error:\n"
                    f"{last_error}\n"
                    f"Please output ONLY a valid JSON object with no trailing commas, "
                    f"no comments, and no markdown backticks."
                )

            response, usage = self.llm.generate(
                prompt=prompt,
                system_prompt=DECOMPOSITION_SYSTEM_PROMPT
            )

            try:
                clean_response = self._clean_json(response)
                data = json.loads(clean_response)

                # Convert JSON items to Step objects
                steps = []
                for item in data.get("items", []):
                    steps.append(Step(**item))

                if not steps:
                    raise ValueError("Plan has 0 steps — LLM returned empty items list")

                plan = Plan(
                    original_goal=data.get("original_goal", context.original_request),
                    items=steps
                )
                return plan
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                last_error = str(e)
                logger.warning(f"Plan parse attempt {plan_attempt + 1}/{max_plan_retries} failed: {e}")
                logger.debug(f"Raw LLM response: {response[:500]}")

        # All retries exhausted
        logger.error(f"Failed to generate valid plan after {max_plan_retries} attempts")
        raise PlanningError(
            message=f"Failed to generate valid plan after {max_plan_retries} attempts: {last_error}",
            raw_response=response,
        )

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
