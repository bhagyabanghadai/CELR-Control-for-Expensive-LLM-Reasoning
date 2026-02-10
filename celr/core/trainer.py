import json
import logging
import os
from typing import List, Dict

from celr.training.scorer import TrajectoryScorer
from celr.training.data_generator import PreferencePairGenerator

logger = logging.getLogger(__name__)

class Trainer:
    """
    The 'Meta-Learning' component.
    Converts successful reasoning trajectories into SFT (Supervised Fine-Tuning) data
    to train smaller models to imitate the successful plans.
    """
    def __init__(self, log_dir: str = ".celr_logs"):
        self.log_dir = log_dir

    def load_successful_trajectories(self) -> List[Dict]:
        data = []
        if not os.path.exists(self.log_dir):
            return []
            
        for filename in os.listdir(self.log_dir):
            if filename.endswith(".jsonl"):
                path = os.path.join(self.log_dir, filename)
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            record = json.loads(line)
                            if record.get("final_status") == "SUCCESS":
                                data.append(record)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping malformed line in {filename}: {e}")
        return data

    def export_to_sharegpt(self, output_file: str = "celr_finetune_data.json"):
        """
        Exports data in ShareGPT format (common for Axolotl, Llama-Factory, etc.)
        """
        success_data = self.load_successful_trajectories()
        training_examples = []
        
        for record in success_data:
            # Construct the conversation
            conversations = [
                {"from": "system", "value": "You are an expert Planner AI."},
                {"from": "human", "value": f"Goal: {record['original_request']}"},
                # In a real impl, we'd serialize the PLAN structure here as the ideal output
                {"from": "gpt", "value": json.dumps(record['plan'], indent=2)}
            ]
            
            training_examples.append({"conversations": conversations})
            
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(training_examples, f, indent=2)
            
        return len(training_examples)

    def export_to_dpo(self, output_file: str = "celr_dpo_data.json") -> int:
        """
        Export trajectory data as DPO preference pairs.
        High-quality trajectories become 'chosen', low-quality become 'rejected'.
        """
        trajectories = self.load_successful_trajectories()
        # Also load failed ones for rejected examples
        all_trajectories = self._load_all_trajectories()

        scorer = TrajectoryScorer()
        scored = scorer.rank(all_trajectories)

        pair_gen = PreferencePairGenerator()
        pairs = pair_gen.create_dpo_pairs(scored)

        return pair_gen.export_trl_format(pairs, output_file)

    def export_to_spin(self, output_file: str = "celr_spin_data.json") -> int:
        """
        Export trajectory data in SPIN format.
        Real successful trajectories as 'chosen', model re-generations as 'rejected'.
        """
        trajectories = self.load_successful_trajectories()

        # For SPIN, we need the successful trajectories as SFT reference
        sft_data = []
        for record in trajectories:
            sft_data.append({
                "prompt": record.get("original_request", ""),
                "response": json.dumps(record.get("plan", []), indent=2),
            })

        # Create SPIN pairs (at this stage, model_responses would come from
        # re-running the model — for now, this exports the SFT half)
        pair_gen = PreferencePairGenerator()
        return pair_gen.export_sharegpt_format(
            [PreferencePairGenerator._make_sft_pair(d) for d in sft_data if d["response"]],
            output_file,
        ) if sft_data else 0

    def _load_all_trajectories(self) -> List[Dict]:
        """Load ALL trajectories (success + failure) for DPO pair generation."""
        data = []
        if not os.path.exists(self.log_dir):
            return []

        for filename in os.listdir(self.log_dir):
            if filename.endswith(".jsonl"):
                path = os.path.join(self.log_dir, filename)
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            record = json.loads(line)
                            data.append(record)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping malformed line in {filename}: {e}")
        return data
