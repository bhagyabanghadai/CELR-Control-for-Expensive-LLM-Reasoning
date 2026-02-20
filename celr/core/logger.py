import json
import logging
import os
from datetime import datetime
from celr.core.types import TaskContext, Plan

logger = logging.getLogger(__name__)

class TrajectoryLogger:
    def __init__(self, log_dir: str = ".celr_logs"):
        self.log_dir = os.path.abspath(log_dir)
        self.traj_dir = os.path.join(self.log_dir, "Traj")
        os.makedirs(self.traj_dir, exist_ok=True)
        # Unique live file for this session
        self.live_file = os.path.join(self.traj_dir, f"celr_live_{int(datetime.now().timestamp())}.json")

    def _build_data(self, context: TaskContext, plan: Plan, status: str) -> dict:
        """Helper to build standard log packet."""
        return {
            "task_id": context.task_id,
            "timestamp": datetime.now().isoformat(),
            "original_request": context.original_request,
            "status": status,
            "cost_usd": context.current_spread_usd,
            "total_retries": sum(s.retry_count for s in plan.items),
            "steps": [
                {
                    "id": step.id,
                    "description": step.description,
                    "difficulty": step.estimated_difficulty,
                    "status": step.status.value,
                    "retry_count": step.retry_count,
                    "cost_usd": getattr(step, "cost_usd", 0.0),
                    # Capture Nano-Cortex state if available
                    "state_vector": getattr(step, "_last_state_vector", None), 
                }
                for step in plan.items
            ],
            "execution_history": context.execution_history,
            "council_debates": getattr(context, "council_debates", [])  # Phase 9
        }

    def update(self, context: TaskContext, plan: Plan, status: str = "RUNNING"):
        """Write current state to live JSON file for Dashboard."""
        data = self._build_data(context, plan, status)
        with open(self.live_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def save_trajectory(self, context: TaskContext, plan: Plan, final_status: str):
        """
        Saves the full interaction history for future training.
        """
        data = self._build_data(context, plan, final_status)
        
        # 1. Update live file one last time
        with open(self.live_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # 2. Append to daily JSONL (Training Data)
        filename = f"celr_trajectory_{datetime.now().strftime('%Y%m%d')}.jsonl"
        filepath = os.path.join(self.log_dir, filename)
        
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")
            
        context.log(f"Trajectory saved to {filepath}")
        logger.info(f"Trajectory saved: {filepath}")
