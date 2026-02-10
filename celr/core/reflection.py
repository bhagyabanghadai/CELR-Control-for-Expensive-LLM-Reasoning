from celr.core.types import Step, TaskContext
from celr.core.llm import BaseLLMProvider

class SelfReflection:
    def __init__(self, llm: BaseLLMProvider):
        self.llm = llm

    def analyze_failure(self, step: Step, context: TaskContext) -> str:
        """
        Analyzes a failed step and suggests a fix or a better approach.
        """
        prompt = f"""
        FAILURE ANALYSIS
        
        The following step failed:
        Description: {step.description}
        Error/Notes: {step.verification_notes or step.output}
        
        Context History:
        {context.execution_history[-3:]}
        
        Analyze WHY it failed and suggest a fix.
        Be concise.
        """
        
        analysis = self.llm.generate(prompt)
        context.log(f"Reflection on Step {step.id}: {analysis}")
        return analysis

    def should_retry(self, step: Step, attempt_count: int) -> bool:
        """
        Decides if we should retry the step based on the failure analysis and attempt count.
        """
        if attempt_count >= 3:
            return False
            
        # Specific heuristic: if error is "Rate Limit", definitely retry.
        # If error is "Logic Error", retry with reflection.
        return True
