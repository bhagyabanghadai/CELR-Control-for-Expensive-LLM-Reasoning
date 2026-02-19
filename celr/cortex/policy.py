import numpy as np
from enum import Enum, auto
from typing import List, Optional
import random

class CortexAction(Enum):
    """
    The Action Space for the Adaptive Cortex.
    """
    PROCEED = auto()        # Continue normal execution (Default)
    ESCALATE = auto()       # Switch to stronger model
    DEESCALATE = auto()     # Switch to cheaper model
    VERIFY = auto()         # Trigger verification (extra check)
    RETRY_COMPRESSED = auto() # Retry with compressed prompt
    STOP = auto()           # Stop and finalize success
    ABORT = auto()          # Abort execution (Unsafe/Too Expensive)

import torch
import os
import numpy as np
from celr.cortex.model_nano import NanoCortex, NanoConfig

class MetaPolicy:
    """
    The Policy Network (or Rule Set) for the Adaptive Cortex.
    """
    def __init__(self, method: str = "hybrid"): # 'hybrid' = RL with heuristic fallback
        self.method = method
        self.actions = list(CortexAction)
        
        # RL Model
        self.model = None
        # Try to load weights on init if they exist
        self.weights_path = os.path.join(os.getcwd(), "cortex_weights.pt")
        if os.path.exists(self.weights_path):
            self.load_weights(self.weights_path)

    def get_action(self, state: np.ndarray, available_actions: List[CortexAction] = None) -> CortexAction:
        """
        Decides the next action.
        Prioritizes RL model if available. Falls back to Heuristics if not.
        """
        if available_actions is None:
            available_actions = self.actions

        # 1. Try Nano-Cortex (RL)
        if self.model is not None and self.method in ["rl", "hybrid"]:
            try:
                msg = self._get_action_rl(state)
                # Ensure the predicted action is valid/available
                if msg in available_actions:
                    return msg
            except Exception as e:
                print(f"NanoCortex Inference Failed: {e}. Falling back to heuristic.")
        
        # 2. Fallback to Heuristic
        return self._get_action_heuristic(state, available_actions)

    def _get_action_rl(self, state: np.ndarray) -> CortexAction:
        """
        Run inference on the Decision Transformer.
        Input: State vector (8,)
        Output: CortexAction
        """
        self.model.eval()
        with torch.no_grad():
            # Prepare inputs - Single step inference
            # We assume a context length of 1 for now (stateless policy behavior) 
            # or we could maintain a history buffer in the class.
            # For v1 simplicity: Treat as Contextual Bandit (History=1)
            
            s_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # (1, 1, 8)
            
            # Dummy inputs for action/return just to satisfy forward pass shape
            # In a real autoregressive loop, we'd pass history.
            a_dummy = torch.zeros((1, 1), dtype=torch.long)
            r_dummy = torch.ones((1, 1, 1), dtype=torch.float32) # Assume we want Success (Return=1.0)
            t_dummy = torch.zeros((1, 1), dtype=torch.long)

            logits = self.model(s_tensor, a_dummy, r_dummy, t_dummy) # (1, 1, act_dim)
            
            # Greedy decoding
            action_idx = torch.argmax(logits[0, -1, :]).item()
            
            # Map index back to Enum
            # We need to correspond this to the training index logic
            # Assuming training used 0-indexed Enum list order
            if 0 <= action_idx < len(self.actions):
                return self.actions[action_idx]
            
            return CortexAction.PROCEED


    def _get_action_heuristic(self, state: np.ndarray, available_actions: List[CortexAction]) -> CortexAction:
        """
        Original hardcoded logic.
        """
        # Unwrap state vector
        budget_ratio = state[0]
        retry_ratio = state[2]
        difficulty = state[4]

        # 1. Safety/Budget Check
        if budget_ratio < 0.05:
            return CortexAction.ABORT
        
        # 2. Difficulty Check
        if difficulty > 0.8 and retry_ratio > 0.3:
             # Hard task failing -> Escalate
             if CortexAction.ESCALATE in available_actions:
                 return CortexAction.ESCALATE

        # 3. Retry Logic
        if retry_ratio > 0.6:
            # Many retries -> Try compressed prompt or Abort
            if CortexAction.RETRY_COMPRESSED in available_actions:
                return CortexAction.RETRY_COMPRESSED
            if retry_ratio > 0.9:
                return CortexAction.ABORT

        # 4. Default: Proceed
        return CortexAction.PROCEED

    def load_weights(self, path: str):
        try:
            config = NanoConfig(state_dim=8, act_dim=len(self.actions))
            self.model = NanoCortex(config)
            self.model.load_state_dict(torch.load(path))
            # print(f"Loaded NanoCortex from {path}")
        except Exception as e:
            print(f"Failed to load weights: {e}")
            self.model = None

    def save_weights(self, path: str):
        if self.model:
            torch.save(self.model.state_dict(), path)
