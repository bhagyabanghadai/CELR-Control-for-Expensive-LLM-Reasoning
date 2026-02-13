"""
CELR Benchmark Tasks — standardized tasks across difficulty levels.

Each task has:
  - id: unique identifier
  - prompt: the task description
  - difficulty: expected difficulty (easy/medium/hard)
  - category: reasoning, coding, math, general
  - expected_contains: keywords that should appear in a correct answer
  - max_budget: budget limit for this task
"""

BENCHMARK_TASKS = [
    # ==================== EASY (0.1-0.3) ====================
    {
        "id": "easy_math_1",
        "prompt": "What is 15 multiplied by 7? Provide only the numeric answer.",
        "difficulty": "easy",
        "category": "math",
        "expected_contains": ["105"],
        "max_budget": 0.10,
    },
    {
        "id": "easy_code_1",
        "prompt": "Write a Python function called `add` that takes two numbers and returns their sum.",
        "difficulty": "easy",
        "category": "coding",
        "expected_contains": ["def add", "return"],
        "max_budget": 0.10,
    },
    {
        "id": "easy_general_1",
        "prompt": "What is the capital of France? Answer in one word.",
        "difficulty": "easy",
        "category": "general",
        "expected_contains": ["Paris"],
        "max_budget": 0.10,
    },
    {
        "id": "easy_reasoning_1",
        "prompt": "If all cats are animals, and Whiskers is a cat, what can you conclude about Whiskers?",
        "difficulty": "easy",
        "category": "reasoning",
        "expected_contains": ["animal"],
        "max_budget": 0.10,
    },

    # ==================== MEDIUM (0.4-0.6) ====================
    {
        "id": "mid_code_1",
        "prompt": "Write a Python function called `is_palindrome` that takes a string and returns True if it reads the same forwards and backwards (case-insensitive, ignoring spaces).",
        "difficulty": "medium",
        "category": "coding",
        "expected_contains": ["def is_palindrome", "return"],
        "max_budget": 0.20,
    },
    {
        "id": "mid_math_1",
        "prompt": "A store has a 25% off sale. If an item costs $80, what is the sale price? Show your work.",
        "difficulty": "medium",
        "category": "math",
        "expected_contains": ["60"],
        "max_budget": 0.20,
    },
    {
        "id": "mid_reasoning_1",
        "prompt": "There are 5 houses in a row. The red house is to the left of the green house. The blue house is in the middle. The yellow house is next to the blue house. Where is the white house? List the order of all houses from left to right.",
        "difficulty": "medium",
        "category": "reasoning",
        "expected_contains": ["white"],
        "max_budget": 0.20,
    },
    {
        "id": "mid_code_2",
        "prompt": "Write a Python function called `fibonacci` that returns the nth Fibonacci number using recursion with memoization.",
        "difficulty": "medium",
        "category": "coding",
        "expected_contains": ["def fibonacci", "return"],
        "max_budget": 0.20,
    },

    # ==================== HARD (0.7-0.9) ====================
    {
        "id": "hard_code_1",
        "prompt": "Write a Python class called `RateLimiter` that implements a sliding window rate limiter. It should have a method `allow_request(timestamp)` that returns True if the request is within the limit (max 10 requests per 60 seconds).",
        "difficulty": "hard",
        "category": "coding",
        "expected_contains": ["class RateLimiter", "allow_request", "def"],
        "max_budget": 0.50,
    },
    {
        "id": "hard_reasoning_1",
        "prompt": "A farmer has 100 meters of fencing. He wants to fence a rectangular area along a river (so only 3 sides need fencing). What dimensions maximize the area? Show your mathematical reasoning.",
        "difficulty": "hard",
        "category": "reasoning",
        "expected_contains": ["50", "25"],
        "max_budget": 0.50,
    },
    {
        "id": "hard_math_1",
        "prompt": "Solve: If f(x) = 3x^2 - 12x + 7, find the minimum value of f(x) and the x value where it occurs. Show step-by-step work.",
        "difficulty": "hard",
        "category": "math",
        "expected_contains": ["2", "-5"],
        "max_budget": 0.50,
    },
    {
        "id": "hard_code_2",
        "prompt": "Write a Python function called `merge_sorted_lists` that merges k sorted lists into one sorted list. Use a heap-based approach for O(n log k) complexity. Include type hints.",
        "difficulty": "hard",
        "category": "coding",
        "expected_contains": ["def merge_sorted_lists", "heap", "return"],
        "max_budget": 0.50,
    },
]


def get_tasks_by_difficulty(difficulty: str):
    """Filter tasks by difficulty level."""
    return [t for t in BENCHMARK_TASKS if t["difficulty"] == difficulty]


def get_tasks_by_category(category: str):
    """Filter tasks by category."""
    return [t for t in BENCHMARK_TASKS if t["category"] == category]
