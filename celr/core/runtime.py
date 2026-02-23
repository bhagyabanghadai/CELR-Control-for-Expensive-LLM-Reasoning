import io
import sys
import traceback
from typing import Dict, Any, Tuple

class PersistentRuntime:
    """
    A stateful Python runtime environment that persists variables across executions.
    Mimics a REPL session or Jupyter kernel.
    """
    def __init__(self):
        self.globals: Dict[str, Any] = {}
        self.locals: Dict[str, Any] = {}
        
        # Pre-import common libraries to save time/tokens in steps
        self._exec_setup()

    def _exec_setup(self):
        """Pre-load common modules into the runtime context."""
        setup_code = """
import math
import datetime
import re
import random
"""
        self.execute(setup_code)

    def execute(self, code: str) -> Tuple[str, bool]:
        """
        Execute code in the persistent context.
        
        Args:
            code: The Python code to execute.
            
        Returns:
            (output, success): 
                - output: The combined stdout/stderr of the execution.
                - success: True if executed without exception, False otherwise.
        """
        # Capture stdout/stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        
        success = True
        
        try:
            # We use the same globals/locals dicts to persist state
            exec(code, self.globals, self.locals)
        except Exception:
            traceback.print_exc()
            success = False
        finally:
            # Restore stdout/stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            
        output = stdout_capture.getvalue() + stderr_capture.getvalue()
        return output, success

    def get_context_snapshot(self) -> str:
        """Returns a string summary of current variables (for debug/LLM context)."""
        # Filter out modules and dunder methods to keep context clean
        snapshot = []
        for k, v in self.locals.items():
            if k.startswith("_"): continue
            if hasattr(v, "__module__") and v.__module__ == "builtins": continue # Skip modules
            if isinstance(v, type(__builtins__)): continue # Skip imports
            
            # Truncate long values
            val_str = str(v)
            if len(val_str) > 100:
                val_str = val_str[:100] + "..."
            snapshot.append(f"{k} = {val_str}")
            
        return "\n".join(snapshot)
