"""Tests for celr.core.tools — ToolRegistry safety and execution."""

import pytest
from celr.core.tools import ToolRegistry, SAFE_BUILTINS
from celr.core.exceptions import ToolExecutionError


class TestToolRegistry:
    def test_register_and_list(self, tool_registry):
        """Default tools should be registered."""
        tools = tool_registry.list_tools()
        assert "python_repl" in tools
        assert "shell_exec" in tools

    def test_register_custom_tool(self, tool_registry):
        tool_registry.register("custom", lambda x: f"Result: {x}")
        assert "custom" in tool_registry.list_tools()

    def test_execute_missing_tool_raises(self, tool_registry):
        with pytest.raises(ToolExecutionError, match="not found"):
            tool_registry.execute("nonexistent_tool")


class TestPythonREPL:
    def test_basic_execution(self, tool_registry):
        result = tool_registry.execute("python_repl", code="print(2 + 2)")
        assert "4" in result

    def test_no_output(self, tool_registry):
        result = tool_registry.execute("python_repl", code="x = 42")
        assert result == "Success (No Output)"

    def test_import_blocked(self, tool_registry):
        """import should fail because __import__ is not in SAFE_BUILTINS."""
        with pytest.raises(ToolExecutionError, match="execution error"):
            tool_registry.execute("python_repl", code="import os")

    def test_open_blocked(self, tool_registry):
        """open() is not in SAFE_BUILTINS so file operations should fail."""
        with pytest.raises(ToolExecutionError):
            tool_registry.execute("python_repl", code="open('test.txt', 'w')")

    def test_safe_builtins_available(self, tool_registry):
        """Basic builtins like len, range should work."""
        result = tool_registry.execute("python_repl", code="print(len(range(10)))")
        assert "10" in result

    def test_syntax_error_raises(self, tool_registry):
        with pytest.raises(ToolExecutionError):
            tool_registry.execute("python_repl", code="def incomplete(:")


class TestSafeBuiltins:
    def test_dangerous_not_present(self):
        """Dangerous builtins should not be in the whitelist."""
        assert "open" not in SAFE_BUILTINS
        assert "__import__" not in SAFE_BUILTINS
        assert "exec" not in SAFE_BUILTINS
        assert "eval" not in SAFE_BUILTINS
        assert "compile" not in SAFE_BUILTINS

    def test_safe_builtins_present(self):
        """Common safe builtins should be available."""
        assert "len" in SAFE_BUILTINS
        assert "range" in SAFE_BUILTINS
        assert "print" in SAFE_BUILTINS
        assert "str" in SAFE_BUILTINS
        assert "int" in SAFE_BUILTINS
        assert "list" in SAFE_BUILTINS
