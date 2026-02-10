from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import uuid
from datetime import datetime

class TaskStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"

class StepType(str, Enum):
    REASONING = "REASONING"  # Pure thought/planning
    EXECUTION = "EXECUTION"  # Tool use
    VERIFICATION = "VERIFICATION" # Checking result

class Step(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    step_type: StepType = StepType.REASONING
    dependencies: List[str] = Field(default_factory=list) # IDs of steps that must finish first
    
    # Execution details
    assigned_agent: Optional[str] = None # e.g., "local-small", "cloud-large"
    output: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    
    # Metadata
    estimated_difficulty: float = 0.5 # 0.0 to 1.0
    cost_usd: float = 0.0
    verification_notes: Optional[str] = None

class Plan(BaseModel):
    items: List[Step]
    original_goal: str
    created_at: datetime = Field(default_factory=datetime.now)

    def get_runnable_steps(self) -> List[Step]:
        """Return steps that are PENDING and have all dependencies met."""
        completed_ids = {s.id for s in self.items if s.status == TaskStatus.COMPLETED}
        runnable = []
        for step in self.items:
            if step.status == TaskStatus.PENDING:
                if all(dep in completed_ids for dep in step.dependencies):
                    runnable.append(step)
        return runnable

class TaskContext(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_request: str
    budget_limit_usd: float
    current_spread_usd: float = 0.0
    
    # Global memory/state
    shared_state: Dict[str, Any] = Field(default_factory=dict)
    execution_history: List[str] = Field(default_factory=list) # Log of events
    
    def log(self, message: str):
        ts = datetime.now().isoformat()
        self.execution_history.append(f"[{ts}] {message}")
        
    @property
    def budget_remaining(self) -> float:
        return self.budget_limit_usd - self.current_spread_usd

class ModelConfig(BaseModel):
    """Configuration for a specific LLM backend."""
    name: str # e.g. "gpt-4o", "ollama/llama3"
    provider: str # "openai", "ollama", "anthropic"
    cost_per_million_input_tokens: float = 0.0
    cost_per_million_output_tokens: float = 0.0
    context_window: int = 4096
    
    # Capabilities
    supports_tools: bool = False
    is_reasoning_model: bool = False # e.g. o1
