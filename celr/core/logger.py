import json
import logging
import os
from datetime import datetime
from celr.core.types import TaskContext, Plan

logger = logging.getLogger(__name__)

class TrajectoryLogger:
    def __init__(self, log_dir: str = ".celr_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def save_trajectory(self, context: TaskContext, plan: Plan, final_status: str):
        """
        Saves the full interaction history for future training.
        """
        # 1. Structure the data
        # We want a format that is easy to convert to SFT (Supervised Fine-Tuning) data later.
        data = {
            "task_id": context.task_id,
            "timestamp": datetime.now().isoformat(),
            "original_request": context.original_request,
            "final_status": final_status,
            "total_cost": context.current_spread_usd,
            "plan": [
                {
                    "id": step.id,
                    "description": step.description,
                    "difficulty": step.estimated_difficulty,
                    "agent": step.assigned_agent,
                    "output": step.output,
                    "verification": step.verification_notes
                }
                for step in plan.items
            ],
            "execution_log": context.execution_history
        }

        # 2. Save to JSONL (one file per day or single append)
        filename = f"celr_trajectory_{datetime.now().strftime('%Y%m%d')}.jsonl"
        filepath = os.path.join(self.log_dir, filename)
        
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")
            
        context.log(f"Trajectory saved to {filepath}")
        logger.info(f"Trajectory saved: {filepath}")
