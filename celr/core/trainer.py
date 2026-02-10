import json
import os
from typing import List, Dict

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
                        except:
                            pass
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
