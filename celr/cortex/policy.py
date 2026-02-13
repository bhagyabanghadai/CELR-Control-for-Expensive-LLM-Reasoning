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

class MetaPolicy:
    """
    The Policy Network (or Rule Set) for the Adaptive Cortex.
    """
    def __init__(self, method: str = "heuristic"):
        self.method = method
        self.actions = list(CortexAction)

    def get_action(self, state: np.ndarray, available_actions: List[CortexAction] = None) -> CortexAction:
        """
        Decides the next action based on the state vector.
        Currently implements the 'Heuristic Baseline' (Cold Start).
        Future versions will load an RL policy.
        """
        # Unwrap state vector
        budget_ratio = state[0]
        retry_ratio = state[2]
        difficulty = state[4]
        
        # Action Masking (Default: All actions available unless specified)
        if available_actions is None:
            available_actions = self.actions

        # Heuristic Logic (Behavior Cloning Baseline)
        
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
        """Placeholder for loading RL weights"""
        pass

    def save_weights(self, path: str):
        """Placeholder for saving RL weights"""
        pass
