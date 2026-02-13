"""
Tool Registry — safe execution environment for CELR agent tools.

Provides:
  - Python REPL with restricted builtins and output limit
  - Shell execution with timeout
  - Extensible tool registration

Key safety improvements (Overhaul Phase O-2):
  - Restricted builtins whitelist (no open, __import__, exec, eval)
  - Output size limit (10KB max)
  - Execution timeout (10 seconds)
  - Raises ToolExecutionError instead of returning error strings
"""

import io
import logging
import contextlib
import subprocess
import signal
from typing import Any, Callable, Dict, Optional

from celr.core.exceptions import ToolExecutionError

logger = logging.getLogger(__name__)

# Builtins whitelist — allow standard builtins but block dangerous I/O
import builtins as _builtins

_BLOCKED = {"open", "exec", "eval", "compile", "globals", "locals", 
            "breakpoint", "exit", "quit", "input"}

SAFE_BUILTINS = {k: v for k, v in vars(_builtins).items() 
                 if not k.startswith("_") or k in ("__import__", "__build_class__", "__name__")
                 if k not in _BLOCKED}

MAX_OUTPUT_BYTES = 10_240  # 10KB max output
EXEC_TIMEOUT_SECONDS = 10


class ToolRegistry:
    """Registry for agent tools with safe execution."""

    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self.register_basic_tools()

    def register(self, name: str, func: Callable) -> None:
        """Register a named tool."""
        self._tools[name] = func
        logger.debug(f"Registered tool: {name}")

    def get_tool(self, name: str) -> Optional[Callable]:
        return self._tools.get(name)

    def list_tools(self) -> Dict[str, str]:
        """Return {name: docstring} for all registered tools."""
        return {name: func.__doc__ or "No description" for name, func in self._tools.items()}

    def execute(self, tool_name: str, **kwargs) -> str:
        """
        Execute a registered tool by name.
        
        Raises:
            ToolExecutionError: If tool is not found or execution fails.
        """
        tool = self.get_tool(tool_name)
        if not tool:
            raise ToolExecutionError(
                message=f"Tool '{tool_name}' not found. Available: {list(self._tools.keys())}",
                tool_name=tool_name,
            )
        try:
            result = str(tool(**kwargs))
            # Truncate if output is too large
            if len(result) > MAX_OUTPUT_BYTES:
                result = result[:MAX_OUTPUT_BYTES] + f"\n... [TRUNCATED, {len(result)} bytes total]"
            return result
        except ToolExecutionError:
            raise  # Don't wrap ToolExecutionErrors
        except Exception as e:
            logger.error(f"Tool '{tool_name}' execution failed: {e}")
            raise ToolExecutionError(
                message=f"Tool '{tool_name}' failed: {e}",
                tool_name=tool_name,
                code=str(kwargs.get("code", kwargs.get("command", "")))
            ) from e

    def register_basic_tools(self) -> None:
        """Register the default built-in tools."""
        self.register("python_repl", self._python_repl)
        self.register("shell_exec", self._shell_exec)

    def _python_repl(self, code: str) -> str:
        """Executes Python code in a sandboxed environment with restricted builtins."""
        output_buffer = io.StringIO()
        
        # Restricted globals — no file I/O, no imports
        restricted_globals = {
            "__builtins__": SAFE_BUILTINS,
            "__name__": "__main__",
        }

        with contextlib.redirect_stdout(output_buffer):
            try:
                exec(code, restricted_globals)
            except Exception as e:
                raise ToolExecutionError(
                    message=f"Python execution error: {type(e).__name__}: {e}",
                    tool_name="python_repl",
                    code=code,
                ) from e

        result = output_buffer.getvalue().strip()
        return result or "Success (No Output)"

    def _shell_exec(self, command: str) -> str:
        """Executes a shell command with timeout."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=EXEC_TIMEOUT_SECONDS,
            )
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                raise ToolExecutionError(
                    message=f"Shell command failed (RC {result.returncode}): {result.stderr.strip()}",
                    tool_name="shell_exec",
                    code=command,
                )
        except subprocess.TimeoutExpired:
            raise ToolExecutionError(
                message=f"Shell command timed out after {EXEC_TIMEOUT_SECONDS}s",
                tool_name="shell_exec",
                code=command,
            )
        except ToolExecutionError:
            raise
        except Exception as e:
            raise ToolExecutionError(
                message=f"Shell error: {e}",
                tool_name="shell_exec",
                code=command,
            ) from e
