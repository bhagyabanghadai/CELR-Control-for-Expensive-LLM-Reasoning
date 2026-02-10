from typing import Callable, Dict, Any, Optional
import subprocess
import sys
import io
import contextlib

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self.register_basic_tools()

    def register(self, name: str, func: Callable):
        self._tools[name] = func

    def get_tool(self, name: str) -> Optional[Callable]:
        return self._tools.get(name)

    def list_tools(self) -> Dict[str, str]:
        return {name: func.__doc__ or "No description" for name, func in self._tools.items()}

    def execute(self, tool_name: str, **kwargs) -> str:
        tool = self.get_tool(tool_name)
        if not tool:
            return f"Error: Tool '{tool_name}' not found."
        try:
            return str(tool(**kwargs))
        except Exception as e:
            return f"Error executing tool '{tool_name}': {e}"

    def register_basic_tools(self):
        self.register("python_repl", self._python_repl)
        self.register("shell_exec", self._shell_exec)

    def _python_repl(self, code: str) -> str:
        """Executes Python code in a sandboxed environment (conceptually)."""
        # capture stdout
        output_buffer = io.StringIO()
        with contextlib.redirect_stdout(output_buffer):
            try:
                # Warning: exec() is dangerous. In a real product, use e2b or docker.
                exec(code, {"__name__": "__main__"})
            except Exception as e:
                return f"Execution Error: {e}"
        return output_buffer.getvalue().strip() or "Success (No Output)"

    def _shell_exec(self, command: str) -> str:
        """Executes a shell command."""
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return f"Error (RC {result.returncode}): {result.stderr.strip()}"
        except Exception as e:
            return f"Shell Error: {e}"
