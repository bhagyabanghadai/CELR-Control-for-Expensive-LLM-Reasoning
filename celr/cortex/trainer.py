import json
import os
from typing import List, Dict
from celr.core.config import CELRConfig

class OfflineTrainer:
    """
    Simulates training an RL policy from offline logs.
    In Phase 8, this calculates rewards and aggregates trajectories.
    """
    def __init__(self, config: CELRConfig):
        self.config = config
        self.log_dir = os.path.join(os.getcwd(), "logs", "Traj")

    def load_trajectories(self) -> List[Dict]:
        """Loads all JSON logs from the trajectory directory."""
        trajs = []
        if not os.path.exists(self.log_dir):
            return []
        
        for f in os.listdir(self.log_dir):
            if f.endswith(".json"):
                try:
                    with open(os.path.join(self.log_dir, f), 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        trajs.append(data)
                except Exception as e:
                    print(f"Error loading {f}: {e}")
        return trajs

    def compute_reward(self, trajectory: Dict) -> float:
        """
        R = Success + Correctness - Cost - Time - (100 * Unsafe)
        """
        status = trajectory.get("status", "FAILED")
        cost = trajectory.get("cost", 0.0)
        time_taken = trajectory.get("time_taken", 0.0)
        unsafe_count = trajectory.get("unsafe_count", 0) # Placeholder if we track this
        
        reward = 0.0
        
        # 1. Success Bonus
        if status == "SUCCESS":
            reward += 10.0
        elif status == "FAILED":
            # Massive penalty for failure if budget was used up
            reward -= 5.0 

        # 2. Cost Penalty (Higher cost = lower reward)
        # Normalize: $1.00 cost = -1.0 reward
        reward -= cost 

        # 3. Time Penalty
        # Normalize: 60s = -0.1 reward
        reward -= (time_taken / 600.0)

        # 4. Safety Value Limit
        if unsafe_count > 0:
            reward -= 100.0

        return reward

    def train_step(self):
        """
        Placeholder for the RL update step.
        For now, it just analyzes the logs and prints stats.
        """
        trajs = self.load_trajectories()
        if not trajs:
            print("No logs found for training.")
            return

        total_reward = 0
        for t in trajs:
            r = self.compute_reward(t)
            total_reward += r
        
        avg_reward = total_reward / len(trajs)
        print(f" Analyzed {len(trajs)} trajectories.")
        print(f" Average Reward: {avg_reward:.4f}")
        print(" Policy update simulation complete.")
