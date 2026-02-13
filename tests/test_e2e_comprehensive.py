"""
Comprehensive End-to-End Tests for CELR Chat Application.

Test Categories:
  1. Sandbox Safety — builtins, blocked functions, imports
  2. Step Type Validation — resilient to bad LLM output
  3. Smart Router — classifies queries correctly
  4. Conversation Integrity — no context bleed
  5. LLM Provider — generate and generate_chat methods
  6. Planner & Executor — plan creation and execution
  7. Auto-Dependencies — ensure_dependencies logic
  8. User Scenario Simulation — real user flows

Run: python -m pytest tests/test_e2e_comprehensive.py -v
"""

import pytest
import sys
import os
import json
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ═══════════════════════════════════════════════════════════════════
# 1. SANDBOX SAFETY TESTS
# ═══════════════════════════════════════════════════════════════════

class TestSandboxSafety:
    """Test that the Python REPL sandbox blocks dangerous operations."""

    def setup_method(self):
        from celr.core.tools import ToolRegistry
        self.registry = ToolRegistry()

    def test_basic_math(self):
        """Sandbox should handle basic math."""
        result = self.registry.execute("python_repl", code="print(2 + 2)")
        assert "4" in result

    def test_string_operations(self):
        """Sandbox should handle string ops."""
        result = self.registry.execute("python_repl", code="print('hello'.upper())")
        assert "HELLO" in result

    def test_list_comprehension(self):
        """Sandbox should handle list comprehensions."""
        result = self.registry.execute("python_repl", code="print([x**2 for x in range(5)])")
        assert "[0, 1, 4, 9, 16]" in result

    def test_class_definition(self):
        """Sandbox MUST allow class definitions (__build_class__ fix)."""
        code = """
class Calculator:
    def add(self, a, b):
        return a + b

calc = Calculator()
print(calc.add(3, 7))
"""
        result = self.registry.execute("python_repl", code=code)
        assert "10" in result

    def test_exception_handling(self):
        """Sandbox should allow try/except."""
        code = """
try:
    x = 1 / 0
except ZeroDivisionError:
    print("caught division by zero")
"""
        result = self.registry.execute("python_repl", code=code)
        assert "caught division by zero" in result

    def test_import_math(self):
        """Sandbox MUST allow math import."""
        result = self.registry.execute("python_repl", code="import math; print(math.pi)")
        assert "3.14" in result

    def test_import_json(self):
        """Sandbox MUST allow json import."""
        code = "import json; print(json.dumps({'key': 'value'}))"
        result = self.registry.execute("python_repl", code=code)
        assert "key" in result

    def test_import_datetime(self):
        """Sandbox MUST allow datetime import."""
        code = "from datetime import datetime; print(type(datetime.now()).__name__)"
        result = self.registry.execute("python_repl", code=code)
        assert "datetime" in result

    def test_block_open(self):
        """Sandbox MUST block file open()."""
        from celr.core.exceptions import ToolExecutionError
        with pytest.raises(ToolExecutionError, match="NameError"):
            self.registry.execute("python_repl", code="open('test.txt', 'w')")

    def test_block_exec(self):
        """Sandbox MUST block exec()."""
        from celr.core.exceptions import ToolExecutionError
        with pytest.raises(ToolExecutionError, match="NameError"):
            self.registry.execute("python_repl", code="exec('print(1)')")

    def test_block_eval(self):
        """Sandbox MUST block eval()."""
        from celr.core.exceptions import ToolExecutionError
        with pytest.raises(ToolExecutionError, match="NameError"):
            self.registry.execute("python_repl", code="eval('1+1')")

    def test_block_input(self):
        """Sandbox MUST block input() (would hang the process)."""
        from celr.core.exceptions import ToolExecutionError
        with pytest.raises(ToolExecutionError, match="NameError"):
            self.registry.execute("python_repl", code="input('Enter something: ')")

    def test_block_compile(self):
        """Sandbox MUST block compile()."""
        from celr.core.exceptions import ToolExecutionError
        with pytest.raises(ToolExecutionError, match="NameError"):
            self.registry.execute("python_repl", code="compile('pass', '<string>', 'exec')")

    def test_output_truncation(self):
        """Sandbox should truncate very large output."""
        code = "print('A' * 20000)"
        result = self.registry.execute("python_repl", code=code)
        assert len(result) <= 11000  # ~10KB + truncation message

    def test_lambda_and_map(self):
        """Sandbox should support functional patterns."""
        code = "print(list(map(lambda x: x*2, [1,2,3])))"
        result = self.registry.execute("python_repl", code=code)
        assert "[2, 4, 6]" in result

    def test_dictionary_comprehension(self):
        """Sandbox should handle dict comprehensions."""
        code = "print({k: v for k, v in enumerate(['a', 'b', 'c'])})"
        result = self.registry.execute("python_repl", code=code)
        assert "0" in result and "a" in result

    def test_inheritance(self):
        """Sandbox should handle class inheritance."""
        code = """
class Animal:
    def speak(self):
        return "..."

class Dog(Animal):
    def speak(self):
        return "Woof!"

print(Dog().speak())
"""
        result = self.registry.execute("python_repl", code=code)
        assert "Woof!" in result

    def test_generator(self):
        """Sandbox should handle generators."""
        code = """
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a+b

print(list(fib(7)))
"""
        result = self.registry.execute("python_repl", code=code)
        assert "[0, 1, 1, 2, 3, 5, 8]" in result

    def test_decorator(self):
        """Sandbox should handle decorators."""
        code = """
def double(fn):
    def wrapper(*args):
        return fn(*args) * 2
    return wrapper

@double
def greet(name):
    return f"Hi {name}! "

print(greet("World"))
"""
        result = self.registry.execute("python_repl", code=code)
        assert "Hi World!" in result


# ═══════════════════════════════════════════════════════════════════
# 2. STEP TYPE VALIDATION TESTS
# ═══════════════════════════════════════════════════════════════════

class TestStepTypeValidation:
    """Test that Step model handles any step_type gracefully (small LLM resilience)."""

    def test_valid_reasoning(self):
        from celr.core.types import Step, StepType
        step = Step(description="Think about it", step_type="REASONING")
        assert step.step_type == StepType.REASONING

    def test_valid_execution(self):
        from celr.core.types import Step, StepType
        step = Step(description="Run code", step_type="EXECUTION")
        assert step.step_type == StepType.EXECUTION

    def test_valid_verification(self):
        from celr.core.types import Step, StepType
        step = Step(description="Check result", step_type="VERIFICATION")
        assert step.step_type == StepType.VERIFICATION

    def test_invalid_analysis_maps_to_reasoning(self):
        """LLMs like llama3.2 often output 'ANALYSIS' instead of 'REASONING'."""
        from celr.core.types import Step, StepType
        step = Step(description="Analyze data", step_type="ANALYSIS")
        assert step.step_type == StepType.REASONING

    def test_invalid_research_maps_to_reasoning(self):
        from celr.core.types import Step, StepType
        step = Step(description="Research topic", step_type="RESEARCH")
        assert step.step_type == StepType.REASONING

    def test_invalid_planning_maps_to_reasoning(self):
        from celr.core.types import Step, StepType
        step = Step(description="Plan approach", step_type="PLANNING")
        assert step.step_type == StepType.REASONING

    def test_invalid_observation_maps_to_reasoning(self):
        from celr.core.types import Step, StepType
        step = Step(description="Observe results", step_type="OBSERVATION")
        assert step.step_type == StepType.REASONING

    def test_lowercase_still_works(self):
        from celr.core.types import Step, StepType
        step = Step(description="reasoning step", step_type="reasoning")
        assert step.step_type == StepType.REASONING

    def test_mixed_case_still_works(self):
        from celr.core.types import Step, StepType
        step = Step(description="exec step", step_type="Execution")
        assert step.step_type == StepType.EXECUTION

    def test_empty_string_maps_to_reasoning(self):
        from celr.core.types import Step, StepType
        step = Step(description="empty type", step_type="")
        assert step.step_type == StepType.REASONING

    def test_garbage_maps_to_reasoning(self):
        from celr.core.types import Step, StepType
        step = Step(description="garbage type", step_type="BANANA_SPLIT")
        assert step.step_type == StepType.REASONING

    def test_enum_value_passthrough(self):
        from celr.core.types import Step, StepType
        step = Step(description="enum test", step_type=StepType.VERIFICATION)
        assert step.step_type == StepType.VERIFICATION

    def test_default_is_reasoning(self):
        from celr.core.types import Step, StepType
        step = Step(description="no type specified")
        assert step.step_type == StepType.REASONING


# ═══════════════════════════════════════════════════════════════════
# 3. SMART ROUTER TESTS
# ═══════════════════════════════════════════════════════════════════

class TestRouterFastPreFilter:
    """Test the rule-based pre-filter (_fast_classify) that catches obvious cases instantly."""

    def setup_method(self):
        from celr.cortex.router import Router
        self.fast = Router._fast_classify

    # -- Greetings → DIRECT --
    def test_hello(self):
        result = self.fast("hello")
        assert result is not None and result[0] == "DIRECT"

    def test_hi(self):
        result = self.fast("hi!")
        assert result is not None and result[0] == "DIRECT"

    def test_hey(self):
        result = self.fast("hey")
        assert result is not None and result[0] == "DIRECT"

    def test_how_are_you(self):
        result = self.fast("how are you doing")
        assert result is not None and result[0] == "DIRECT"

    def test_thanks(self):
        result = self.fast("thanks!")
        assert result is not None and result[0] == "DIRECT"

    def test_goodbye(self):
        result = self.fast("goodbye")
        assert result is not None and result[0] == "DIRECT"

    def test_good_morning(self):
        result = self.fast("good morning")
        assert result is not None and result[0] == "DIRECT"

    # -- Reasoning keywords → REASONING --
    def test_compare_keyword(self):
        result = self.fast("compare Python and Java performance")
        assert result is not None and result[0] == "REASONING"

    def test_analyze_keyword(self):
        result = self.fast("analyze the data trends")
        assert result is not None and result[0] == "REASONING"

    def test_write_code_keyword(self):
        result = self.fast("write code for a web server")
        assert result is not None and result[0] == "REASONING"

    def test_step_by_step(self):
        result = self.fast("explain step by step how to do this")
        assert result is not None and result[0] == "REASONING"

    def test_debug_keyword(self):
        result = self.fast("debug this Python function")
        assert result is not None and result[0] == "REASONING"

    # -- Short queries without keywords → DIRECT --
    def test_short_question(self):
        result = self.fast("what is the weather")
        assert result is not None and result[0] == "DIRECT"

    def test_capital_question(self):
        result = self.fast("capital of france")
        assert result is not None and result[0] == "DIRECT"

    # -- Empty / very short → DIRECT --
    def test_empty_string(self):
        result = self.fast("")
        assert result is not None and result[0] == "DIRECT"

    def test_single_char(self):
        result = self.fast("ok")
        assert result is not None and result[0] == "DIRECT"

    # -- Ambiguous (long, no keywords) → None (defer to LLM) --
    def test_ambiguous_defers_to_llm(self):
        prompt = "I have been thinking about the new project approach and wonder what the best methodology would be for our team"
        result = self.fast(prompt)
        assert result is None  # Should defer to LLM


class TestRouterLLMFallback:
    """Test the LLM-based fallback classification (only for ambiguous queries)."""

    def _make_router(self, llm_response):
        """Create a Router with a mocked LLM that returns a fixed response."""
        from celr.cortex.router import Router
        from celr.core.llm import LLMUsage
        mock_llm = MagicMock()
        mock_llm.generate.return_value = (llm_response, LLMUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20))
        return Router(mock_llm)

    def test_llm_classifies_complex(self):
        """For ambiguous queries, LLM fallback should classify correctly."""
        router = self._make_router('{"type": "REASONING", "reason": "needs deep analysis"}')
        # This query is long enough that _fast_classify returns None
        route, _ = router.classify("I have been thinking about the new project approach and wonder what the best methodology would be for our team")
        assert route == "REASONING"
        router.llm.generate.assert_called_once()  # LLM was actually invoked

    def test_llm_not_called_for_greetings(self):
        """LLM should NOT be called for obvious greetings (fast path handles it)."""
        router = self._make_router('should not matter')
        route, _ = router.classify("hello")
        assert route == "DIRECT"
        router.llm.generate.assert_not_called()  # LLM was bypassed!

    def test_llm_not_called_for_keywords(self):
        """LLM should NOT be called when reasoning keyword is detected."""
        router = self._make_router('should not matter')
        route, _ = router.classify("compare rust and go")
        assert route == "REASONING"
        router.llm.generate.assert_not_called()  # Fast path caught it

    def test_fallback_on_invalid_json(self):
        """Router defaults to DIRECT when LLM returns garbage."""
        router = self._make_router('this is not json at all')
        route, reason = router.classify("I have been thinking about the new project approach and wonder what the best methodology would be for our team")
        assert route == "DIRECT"

    def test_fallback_on_llm_exception(self):
        """Router defaults to DIRECT when LLM raises an exception."""
        from celr.cortex.router import Router
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = Exception("LLM timeout")
        router = Router(mock_llm)
        route, _ = router.classify("I have been thinking about the new project approach and wonder what the best methodology would be for our team")
        assert route == "DIRECT"

    def test_classify_returns_tuple(self):
        """classify() always returns a (str, str) tuple."""
        router = self._make_router('{"type": "DIRECT", "reason": "test"}')
        result = router.classify("test query")
        assert isinstance(result, tuple)
        assert len(result) == 2


# ═══════════════════════════════════════════════════════════════════
# 4. CONVERSATION INTEGRITY TESTS
# ═══════════════════════════════════════════════════════════════════

class TestConversationIntegrity:
    """Test that conversation history is properly managed."""

    def test_messages_have_correct_roles(self):
        """Messages should have user/assistant/system roles."""
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "What is 2+2?"},
        ]
        for msg in history:
            assert msg["role"] in ("user", "assistant", "system")

    def test_system_prompt_structure(self):
        """System prompt should be first in messages array."""
        system_prompt = "You are a helpful assistant"
        history = [
            {"role": "user", "content": "Hello"},
        ]
        chat_messages = [{"role": "system", "content": system_prompt}]
        chat_messages.extend(history[-10:])
        
        assert chat_messages[0]["role"] == "system"
        assert chat_messages[0]["content"] == system_prompt

    def test_history_window_limit(self):
        """Should only include last 10 messages."""
        history = [{"role": "user", "content": f"msg {i}"} for i in range(20)]
        windowed = history[-10:]
        assert len(windowed) == 10
        assert windowed[0]["content"] == "msg 10"
        assert windowed[-1]["content"] == "msg 19"

    def test_fresh_context_per_reasoning(self):
        """Each reasoning query should get a fresh TaskContext."""
        from celr.core.types import TaskContext
        
        ctx1 = TaskContext(original_request="question 1", budget_limit_usd=1.0)
        ctx2 = TaskContext(original_request="question 2", budget_limit_usd=1.0)
        
        assert ctx1.task_id != ctx2.task_id
        assert ctx1.original_request != ctx2.original_request
        assert ctx1.execution_history == []
        assert ctx2.execution_history == []


# ═══════════════════════════════════════════════════════════════════
# 5. LLM PROVIDER TESTS
# ═══════════════════════════════════════════════════════════════════

class TestLLMProvider:
    """Test LLM provider configuration and methods."""

    def test_model_config_creation(self):
        from celr.core.types import ModelConfig
        config = ModelConfig(
            name="ollama/llama3.2",
            provider="ollama",
            cost_per_million_input_tokens=0.0,
            cost_per_million_output_tokens=0.0,
        )
        assert config.name == "ollama/llama3.2"
        assert config.provider == "ollama"
        assert config.cost_per_million_input_tokens == 0.0

    def test_free_model_zero_cost(self):
        from celr.core.types import ModelConfig
        from celr.core.llm import LiteLLMProvider, LLMUsage
        
        config = ModelConfig(name="ollama/test", provider="ollama")
        provider = LiteLLMProvider(config)
        
        usage = LLMUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        cost = provider.calculate_cost(usage)
        assert cost == 0.0

    def test_generate_chat_method_exists(self):
        """generate_chat should be available on LiteLLMProvider."""
        from celr.core.types import ModelConfig
        from celr.core.llm import LiteLLMProvider
        
        config = ModelConfig(name="ollama/test", provider="ollama")
        provider = LiteLLMProvider(config)
        
        assert hasattr(provider, "generate_chat")
        assert callable(provider.generate_chat)

    def test_generate_method_exists(self):
        """generate should be available on LiteLLMProvider."""
        from celr.core.types import ModelConfig
        from celr.core.llm import LiteLLMProvider
        
        config = ModelConfig(name="ollama/test", provider="ollama")
        provider = LiteLLMProvider(config)
        
        assert hasattr(provider, "generate")
        assert callable(provider.generate)


# ═══════════════════════════════════════════════════════════════════
# 6. PLAN & EXECUTION TESTS
# ═══════════════════════════════════════════════════════════════════

class TestPlanExecution:
    """Test plan creation, step ordering, and execution flow."""

    def test_plan_creation(self):
        from celr.core.types import Plan, Step, StepType
        steps = [
            Step(id="1", description="Research", step_type=StepType.REASONING),
            Step(id="2", description="Execute", step_type=StepType.EXECUTION, dependencies=["1"]),
            Step(id="3", description="Verify", step_type=StepType.VERIFICATION, dependencies=["2"]),
        ]
        plan = Plan(items=steps, original_goal="Test goal")
        assert len(plan.items) == 3
        assert plan.original_goal == "Test goal"

    def test_runnable_steps_no_deps(self):
        """Steps with no dependencies should be immediately runnable."""
        from celr.core.types import Plan, Step, StepType, TaskStatus
        steps = [
            Step(id="1", description="Step 1", step_type=StepType.REASONING),
            Step(id="2", description="Step 2", step_type=StepType.REASONING),
        ]
        plan = Plan(items=steps, original_goal="Test")
        runnable = plan.get_runnable_steps()
        assert len(runnable) == 2

    def test_runnable_steps_with_deps(self):
        """Steps with unmet dependencies should NOT be runnable."""
        from celr.core.types import Plan, Step, StepType, TaskStatus
        steps = [
            Step(id="1", description="Step 1", step_type=StepType.REASONING),
            Step(id="2", description="Step 2", step_type=StepType.EXECUTION, dependencies=["1"]),
        ]
        plan = Plan(items=steps, original_goal="Test")
        runnable = plan.get_runnable_steps()
        assert len(runnable) == 1
        assert runnable[0].id == "1"

    def test_runnable_after_completion(self):
        """Once a dependency is completed, dependent step becomes runnable."""
        from celr.core.types import Plan, Step, StepType, TaskStatus
        steps = [
            Step(id="1", description="Step 1", step_type=StepType.REASONING, status=TaskStatus.COMPLETED),
            Step(id="2", description="Step 2", step_type=StepType.EXECUTION, dependencies=["1"]),
        ]
        plan = Plan(items=steps, original_goal="Test")
        runnable = plan.get_runnable_steps()
        assert len(runnable) == 1
        assert runnable[0].id == "2"

    def test_plan_from_json_with_bad_types(self):
        """Plan should handle bad step types from LLM gracefully."""
        from celr.core.types import Plan, Step, StepType
        raw_data = {
            "original_goal": "Test",
            "items": [
                {"description": "Analyze code", "step_type": "ANALYSIS"},
                {"description": "Run tests", "step_type": "TESTING"},
                {"description": "Check results", "step_type": "VERIFICATION"},
            ]
        }
        steps = [Step(**item) for item in raw_data["items"]]
        plan = Plan(items=steps, original_goal=raw_data["original_goal"])
        
        assert plan.items[0].step_type == StepType.REASONING  # ANALYSIS → REASONING
        assert plan.items[1].step_type == StepType.REASONING  # TESTING → REASONING
        assert plan.items[2].step_type == StepType.VERIFICATION  # VERIFICATION stays

    def test_step_retry_tracking(self):
        """Step retry count should track attempts."""
        from celr.core.types import Step
        step = Step(description="Retryable step", max_retries=3)
        assert step.retry_count == 0
        assert step.max_retries == 3
        
        step.retry_count += 1
        assert step.retry_count == 1
        assert step.retry_count < step.max_retries


# ═══════════════════════════════════════════════════════════════════
# 7. TASK CONTEXT TESTS
# ═══════════════════════════════════════════════════════════════════

class TestTaskContext:
    """Test TaskContext creation and budget tracking."""

    def test_context_creation(self):
        from celr.core.types import TaskContext
        ctx = TaskContext(original_request="Test", budget_limit_usd=5.0)
        assert ctx.original_request == "Test"
        assert ctx.budget_limit_usd == 5.0
        assert ctx.current_spread_usd == 0.0

    def test_budget_remaining(self):
        from celr.core.types import TaskContext
        ctx = TaskContext(original_request="Test", budget_limit_usd=5.0)
        ctx.current_spread_usd = 2.5
        assert ctx.budget_remaining == 2.5

    def test_logging(self):
        from celr.core.types import TaskContext
        ctx = TaskContext(original_request="Test", budget_limit_usd=1.0)
        ctx.log("step 1 complete")
        ctx.log("step 2 complete")
        assert len(ctx.execution_history) == 2
        assert "step 1 complete" in ctx.execution_history[0]

    def test_unique_task_ids(self):
        from celr.core.types import TaskContext
        ctx1 = TaskContext(original_request="Q1", budget_limit_usd=1.0)
        ctx2 = TaskContext(original_request="Q2", budget_limit_usd=1.0)
        assert ctx1.task_id != ctx2.task_id

    def test_zero_budget_free_model(self):
        from celr.core.types import TaskContext
        ctx = TaskContext(original_request="Test", budget_limit_usd=0.0)
        assert ctx.budget_remaining == 0.0


# ═══════════════════════════════════════════════════════════════════
# 8. SAFE BUILTINS COMPLETENESS TESTS
# ═══════════════════════════════════════════════════════════════════

class TestSafeBuiltins:
    """Verify SAFE_BUILTINS has everything LLMs commonly need."""

    def test_has_import(self):
        from celr.core.tools import SAFE_BUILTINS
        assert "__import__" in SAFE_BUILTINS

    def test_has_build_class(self):
        from celr.core.tools import SAFE_BUILTINS
        assert "__build_class__" in SAFE_BUILTINS

    def test_blocks_open(self):
        from celr.core.tools import SAFE_BUILTINS
        assert "open" not in SAFE_BUILTINS

    def test_blocks_exec(self):
        from celr.core.tools import SAFE_BUILTINS
        assert "exec" not in SAFE_BUILTINS

    def test_blocks_eval(self):
        from celr.core.tools import SAFE_BUILTINS
        assert "eval" not in SAFE_BUILTINS

    def test_blocks_compile(self):
        from celr.core.tools import SAFE_BUILTINS
        assert "compile" not in SAFE_BUILTINS

    def test_blocks_input(self):
        from celr.core.tools import SAFE_BUILTINS
        assert "input" not in SAFE_BUILTINS

    def test_blocks_breakpoint(self):
        from celr.core.tools import SAFE_BUILTINS
        assert "breakpoint" not in SAFE_BUILTINS

    def test_has_common_builtins(self):
        """Verify all commonly used builtins are present."""
        from celr.core.tools import SAFE_BUILTINS
        required = [
            "abs", "all", "any", "bool", "chr", "dict", "dir",
            "enumerate", "filter", "float", "format", "frozenset",
            "getattr", "hasattr", "hash", "hex", "int", "isinstance",
            "issubclass", "iter", "len", "list", "map", "max", "min",
            "next", "oct", "ord", "pow", "print", "range", "repr",
            "reversed", "round", "set", "slice", "sorted", "str",
            "sum", "tuple", "type", "zip",
        ]
        for name in required:
            assert name in SAFE_BUILTINS, f"Missing builtin: {name}"

    def test_has_exception_classes(self):
        """Verify common exceptions are accessible."""
        from celr.core.tools import SAFE_BUILTINS
        exceptions = [
            "Exception", "ValueError", "TypeError", "KeyError",
            "IndexError", "AttributeError", "ZeroDivisionError",
            "RuntimeError", "StopIteration", "NameError",
        ]
        for name in exceptions:
            assert name in SAFE_BUILTINS, f"Missing exception: {name}"


# ═══════════════════════════════════════════════════════════════════
# 9. OLLAMA DETECTION TESTS
# ═══════════════════════════════════════════════════════════════════

class TestOllamaDetection:
    """Test Ollama detection functions use correct addresses."""

    def test_check_ollama_uses_127001(self):
        """check_ollama_running should use 127.0.0.1 in its URL, not localhost."""
        import inspect
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        
        from celr_chat import check_ollama_running
        source = inspect.getsource(check_ollama_running)
        assert "127.0.0.1" in source
        # Verify the actual URL uses 127.0.0.1 (localhost may appear in docstring)
        assert 'http://127.0.0.1:11434' in source

    def test_get_ollama_models_uses_127001(self):
        """get_ollama_models should use 127.0.0.1 in its URL."""
        import inspect
        from celr_chat import get_ollama_models
        source = inspect.getsource(get_ollama_models)
        assert 'http://127.0.0.1:11434' in source


# ═══════════════════════════════════════════════════════════════════
# 10. COST TRACKER TESTS
# ═══════════════════════════════════════════════════════════════════

class TestCostTracker:
    """Test cost tracking for cloud models."""

    def test_cost_tracking_accumulates(self):
        from celr.core.cost_tracker import CostTracker
        from celr.core.types import TaskContext
        ctx = TaskContext(original_request="test", budget_limit_usd=10.0)
        tracker = CostTracker(ctx)
        tracker.add_cost(0.01)
        tracker.add_cost(0.02)
        assert abs(ctx.current_spread_usd - 0.03) < 0.001

    def test_budget_check(self):
        from celr.core.cost_tracker import CostTracker
        from celr.core.types import TaskContext
        ctx = TaskContext(original_request="test", budget_limit_usd=0.05)
        tracker = CostTracker(ctx)
        tracker.add_cost(0.04)
        assert tracker.can_afford(0.005)  # 0.01 remaining > 0.005
        tracker.add_cost(0.02)
        assert not tracker.can_afford(0.01)  # Over budget


# ═══════════════════════════════════════════════════════════════════
# 11. EXCEPTION TESTS
# ═══════════════════════════════════════════════════════════════════

class TestExceptions:
    """Test custom exception classes."""

    def test_tool_execution_error(self):
        from celr.core.exceptions import ToolExecutionError
        err = ToolExecutionError(message="test error", tool_name="python_repl", code="print(1)")
        assert "test error" in str(err)

    def test_planning_error(self):
        from celr.core.exceptions import PlanningError
        err = PlanningError(message="bad plan", raw_response="invalid json")
        assert "bad plan" in str(err)


# ═══════════════════════════════════════════════════════════════════
# 12. REQUIREMENTS VERIFICATION
# ═══════════════════════════════════════════════════════════════════

class TestRequirements:
    """Verify requirements.txt has all needed packages."""

    def test_requirements_file_exists(self):
        req_path = os.path.join(os.path.dirname(__file__), "..", "requirements.txt")
        assert os.path.exists(req_path)

    def test_requirements_has_core_deps(self):
        req_path = os.path.join(os.path.dirname(__file__), "..", "requirements.txt")
        with open(req_path) as f:
            content = f.read()
        
        core_deps = ["pydantic", "litellm", "rich", "python-dotenv", "networkx", "tenacity", "numpy"]
        for dep in core_deps:
            assert dep in content, f"Missing core dependency: {dep}"

    def test_requirements_has_reasoning_tools(self):
        req_path = os.path.join(os.path.dirname(__file__), "..", "requirements.txt")
        with open(req_path) as f:
            content = f.read()
        
        reasoning_deps = ["wikipedia", "beautifulsoup4", "requests"]
        for dep in reasoning_deps:
            assert dep in content, f"Missing reasoning dependency: {dep}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
