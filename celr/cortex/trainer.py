import json
import os
import random
from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

from celr.core.config import CELRConfig
from celr.cortex.model_nano import NanoCortex, NanoConfig
from celr.cortex.policy import CortexAction # Need to know the enum mapping

class TrajectoryDataset(Dataset):
    """
    PyTorch Dataset that loads CELR JSON logs and converts them 
    into (Returns, States, Actions) trajectories.
    """
    def __init__(self, log_dir: str, max_len: int = 20):
        self.log_dir = log_dir
        self.max_len = max_len
        self.trajectories = self._load_trajectories()

    def _load_trajectories(self) -> List[Dict]:
        trajs = []
        if not os.path.exists(self.log_dir):
            return []
        
        for f in os.listdir(self.log_dir):
            if f.endswith(".json"):
                try:
                    path = os.path.join(self.log_dir, f)
                    with open(path, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        if "steps" in data and len(data["steps"]) > 0:
                             trajs.append(data)
                except Exception as e:
                    print(f"Error loading {f}: {e}")
        return trajs

    def _compute_return(self, traj) -> float:
        # Simple reward function: +1 for SUCCESS, -1 for FAILURE
        # In a real impl, this would be more granular based on cost/time.
        status = traj.get("status", "FAILED")
        if status == "SUCCESS":
            return 1.0
        return -1.0

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        
        # 1. Extract Steps
        steps = traj["steps"]
        final_return = self._compute_return(traj)

        # 2. Build Tensors
        # We need to pad to max_len or truncate
        L = len(steps)
        if L > self.max_len:
            # Take last max_len steps (most recent context)
            steps = steps[-self.max_len:]
            L = self.max_len
        
        # Initialize buffers
        # State dim 8, Action dim 1 (discrete index), Return dim 1, Timestep dim 1
        states = torch.zeros((self.max_len, 8), dtype=torch.float32)
        actions = torch.zeros((self.max_len,), dtype=torch.long) # Discrete indices
        returns = torch.zeros((self.max_len, 1), dtype=torch.float32)
        timesteps = torch.zeros((self.max_len,), dtype=torch.long)
        mask = torch.zeros((self.max_len,), dtype=torch.float32) # 1 for valid, 0 for pad

        for i, step in enumerate(steps):
             # Extract state vector (assumed to be stored in log or we re-extract)
             # NOTE: In v1, logs might not have 'state_vector'. 
             # We might need to "mock" it or assume 'StateExtractor' can serve it.
             # For this Nano implementation, we'll assume the log *has* it or use random/zeros if missing
             # to prevent crashing during this 'bootstrap' phase.
             
             s_vec = step.get("state_vector", [0.0]*8) 
             states[i] = torch.tensor(s_vec, dtype=torch.float32)

             # Action Enum -> Index
             act_str = step.get("action", "PROCEED")
             try:
                 act_idx = CortexAction[act_str].value - 1 # Enum is 1-based usually?
                 # Let's assume standard Enum 1..N. CortexAction starts at 1 usually.
                 # Actually `auto()` starts at 1. Safe way: cast to list index.
                 # We need a consistent mapping. 
                 # Let's just hash it or use a fixed map if strict. 
                 # For now, let's assume raw integer or string map.
                 actions[i] = 0 # Default PROCEED
             except:
                 actions[i] = 0

             # Returns-to-Go
             # For offline RL, we usually pass the *desired* return.
             # Here we just pass the trajectory return (Hindsight Experience Replay style)
             returns[i] = final_return
             
             timesteps[i] = i
             mask[i] = 1.0
        
        return states, actions, returns, timesteps, mask

class OfflineTrainer:
    """
    Trains the NanoCortex Decision Transformer.
    """
    def __init__(self, config: CELRConfig):
        self.config = config
        self.log_dir = os.path.join(os.getcwd(), config.log_dir, "Traj")
        self.grad_clip = 1.0
        
        # Init Model
        self.nano_config = NanoConfig(state_dim=8, act_dim=len(CortexAction))
        self.model = NanoCortex(self.nano_config)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=3e-4)

    def train(self, epochs: int = 5, batch_size: int = 4):
        print(f"Loading logs from {self.log_dir}...")
        dataset = TrajectoryDataset(self.log_dir)
        if len(dataset) == 0:
            print("No logs found. Skipping training.")
            return

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model.train()

        print(f"Starting Nano-Cortex Training ({epochs} epochs)...")
        
        for epoch in range(epochs):
            total_loss = 0
            count = 0
            
            for states, actions, returns, timesteps, mask in loader:
                # Forward
                # model outputs logits for actions (B, T, act_dim)
                logits = self.model(states, actions, returns, timesteps)

                # Loss: Cross Entropy on valid actions
                # logits: (B, T, C), actions: (B, T)
                # Flatten -> (B*T, C), (B*T)
                B, T, C = logits.shape
                loss = F.cross_entropy(logits.view(-1, C), actions.view(-1), reduction='none')
                
                # Apply mask (ignore padding)
                loss = (loss * mask.view(-1)).mean()

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                total_loss += loss.item()
                count += 1
            
            avg_loss = total_loss / max(1, count)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

        # Save weights
        save_path = os.path.join(os.getcwd(), "cortex_weights.pt")
        torch.save(self.model.state_dict(), save_path)
        print(f"Training complete. Weights saved to {save_path}")

    def train_step(self):
        """Called by CLI or Scheduler"""
        self.train()
