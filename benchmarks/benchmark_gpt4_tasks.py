"""
GPT-4 Level Benchmark Tasks for CELR.

20 tasks modeled after real GPT-4 evaluation benchmarks:
  - MMLU (knowledge):     4 multiple-choice questions across subjects
  - HumanEval (code):     4 Python function generation tasks
  - GSM8K (math):         4 grade-school math word problems
  - ARC (science):        4 science reasoning questions
  - MATH (competition):   4 competition-level math problems

Each task has a definitive correct answer for auto-grading,
plus GPT-4's published reference score for that category.

GPT-4 reference scores (OpenAI, March 2023 / updated 2024):
  MMLU:       86.4%
  HumanEval:  67.0%
  GSM8K:      92.0%
  ARC:        96.3%
  MATH:       42.5%
"""

# GPT-4 published reference accuracy per category
GPT4_REFERENCE_SCORES = {
    "mmlu":       86.4,
    "humaneval":  67.0,
    "gsm8k":      92.0,
    "arc":        96.3,
    "math":       42.5,
}

GPT4_BENCHMARK_TASKS = [

    # ═══════════════════════════════════════════════════════════════
    # 1. MMLU — Knowledge & Comprehension (GPT-4: 86.4%)
    # ═══════════════════════════════════════════════════════════════

    {
        "id": "mmlu_history_1",
        "prompt": (
            "Question: Which event directly led to the United States' entry into World War I?\n"
            "A) The assassination of Archduke Franz Ferdinand\n"
            "B) The sinking of the Lusitania\n"
            "C) The Zimmermann Telegram and unrestricted submarine warfare\n"
            "D) The bombing of Pearl Harbor\n\n"
            "Answer with just the letter and a brief explanation."
        ),
        "category": "mmlu",
        "correct_answer": "C",
        "expected_contains": ["C"],
        "max_budget": 0.10,
    },
    {
        "id": "mmlu_biology_1",
        "prompt": (
            "Question: Which organelle is primarily responsible for producing ATP in eukaryotic cells?\n"
            "A) Ribosome\n"
            "B) Golgi apparatus\n"
            "C) Mitochondria\n"
            "D) Endoplasmic reticulum\n\n"
            "Answer with just the letter and a brief explanation."
        ),
        "category": "mmlu",
        "correct_answer": "C",
        "expected_contains": ["C", "mitochondria"],
        "max_budget": 0.10,
    },
    {
        "id": "mmlu_law_1",
        "prompt": (
            "Question: In U.S. constitutional law, which amendment abolished slavery?\n"
            "A) 13th Amendment\n"
            "B) 14th Amendment\n"
            "C) 15th Amendment\n"
            "D) 19th Amendment\n\n"
            "Answer with just the letter and a brief explanation."
        ),
        "category": "mmlu",
        "correct_answer": "A",
        "expected_contains": ["A", "13"],
        "max_budget": 0.10,
    },
    {
        "id": "mmlu_geography_1",
        "prompt": (
            "Question: What is the longest river in the world?\n"
            "A) Amazon River\n"
            "B) Nile River\n"
            "C) Yangtze River\n"
            "D) Mississippi River\n\n"
            "Answer with just the letter and a brief explanation."
        ),
        "category": "mmlu",
        "correct_answer": "B",
        "expected_contains": ["Nile"],
        "max_budget": 0.10,
    },

    # ═══════════════════════════════════════════════════════════════
    # 2. HumanEval — Code Generation (GPT-4: 67.0%)
    # ═══════════════════════════════════════════════════════════════

    {
        "id": "humaneval_1",
        "prompt": (
            "Write a Python function called `two_sum` that takes a list of integers and a target integer. "
            "Return the indices of the two numbers that add up to the target. "
            "Assume exactly one solution exists. Do not use the same element twice.\n\n"
            "Example: two_sum([2, 7, 11, 15], 9) should return [0, 1]"
        ),
        "category": "humaneval",
        "correct_answer": "def two_sum",
        "expected_contains": ["def two_sum", "return"],
        "max_budget": 0.20,
    },
    {
        "id": "humaneval_2",
        "prompt": (
            "Write a Python function called `is_valid_parentheses` that takes a string containing "
            "only '(', ')', '{', '}', '[', ']' and returns True if the input string is valid. "
            "An input string is valid if: open brackets are closed by the same type, "
            "and open brackets are closed in the correct order.\n\n"
            "Examples:\n"
            "  is_valid_parentheses('()[]{}') -> True\n"
            "  is_valid_parentheses('(]') -> False\n"
            "  is_valid_parentheses('([)]') -> False"
        ),
        "category": "humaneval",
        "correct_answer": "def is_valid_parentheses",
        "expected_contains": ["def is_valid_parentheses", "return"],
        "max_budget": 0.20,
    },
    {
        "id": "humaneval_3",
        "prompt": (
            "Write a Python function called `longest_common_prefix` that takes a list of strings "
            "and returns the longest common prefix string amongst them. "
            "If there is no common prefix, return an empty string.\n\n"
            "Examples:\n"
            "  longest_common_prefix(['flower', 'flow', 'flight']) -> 'fl'\n"
            "  longest_common_prefix(['dog', 'racecar', 'car']) -> ''"
        ),
        "category": "humaneval",
        "correct_answer": "def longest_common_prefix",
        "expected_contains": ["def longest_common_prefix", "return"],
        "max_budget": 0.20,
    },
    {
        "id": "humaneval_4",
        "prompt": (
            "Write a Python function called `max_profit` that takes a list of stock prices "
            "(one price per day) and returns the maximum profit from buying and selling once. "
            "If no profit is possible, return 0.\n\n"
            "Examples:\n"
            "  max_profit([7, 1, 5, 3, 6, 4]) -> 5  (buy at 1, sell at 6)\n"
            "  max_profit([7, 6, 4, 3, 1]) -> 0  (no profitable trade)"
        ),
        "category": "humaneval",
        "correct_answer": "def max_profit",
        "expected_contains": ["def max_profit", "return"],
        "max_budget": 0.20,
    },

    # ═══════════════════════════════════════════════════════════════
    # 3. GSM8K — Grade School Math (GPT-4: 92.0%)
    # ═══════════════════════════════════════════════════════════════

    {
        "id": "gsm8k_1",
        "prompt": (
            "A store sells notebooks for $3 each and pens for $1.50 each. "
            "Maria buys 4 notebooks and 6 pens. She pays with a $50 bill. "
            "How much change does she receive? Show your work step by step."
        ),
        "category": "gsm8k",
        "correct_answer": "29",
        "expected_contains": ["29"],
        "max_budget": 0.15,
    },
    {
        "id": "gsm8k_2",
        "prompt": (
            "A train travels at 60 mph for 2.5 hours, then at 80 mph for 1.5 hours. "
            "What is the total distance traveled? Show your work step by step."
        ),
        "category": "gsm8k",
        "correct_answer": "270",
        "expected_contains": ["270"],
        "max_budget": 0.15,
    },
    {
        "id": "gsm8k_3",
        "prompt": (
            "A bakery makes 120 cupcakes. They sell 3/4 of them in the morning at $2.50 each, "
            "and the remaining at half price in the afternoon. "
            "How much total revenue does the bakery make? Show your work step by step."
        ),
        "category": "gsm8k",
        "correct_answer": "262.5",
        "expected_contains": ["262.5", "262.50"],
        "max_budget": 0.15,
    },
    {
        "id": "gsm8k_4",
        "prompt": (
            "Tom has 3 times as many marbles as Jerry. Jerry has 5 more marbles than Sam. "
            "Sam has 8 marbles. How many marbles do they have altogether? "
            "Show your work step by step."
        ),
        "category": "gsm8k",
        "correct_answer": "60",
        "expected_contains": ["60"],
        "max_budget": 0.15,
    },

    # ═══════════════════════════════════════════════════════════════
    # 4. ARC — Science Reasoning (GPT-4: 96.3%)
    # ═══════════════════════════════════════════════════════════════

    {
        "id": "arc_physics_1",
        "prompt": (
            "Question: A ball is thrown straight up into the air. At the highest point of its flight, "
            "what is its velocity and acceleration?\n"
            "A) Velocity is zero, acceleration is zero\n"
            "B) Velocity is zero, acceleration is 9.8 m/s² downward\n"
            "C) Velocity is maximum, acceleration is zero\n"
            "D) Velocity is maximum, acceleration is 9.8 m/s² downward\n\n"
            "Answer with just the letter and a brief explanation."
        ),
        "category": "arc",
        "correct_answer": "B",
        "expected_contains": ["B"],
        "max_budget": 0.10,
    },
    {
        "id": "arc_chemistry_1",
        "prompt": (
            "Question: What happens to the mass of iron when it rusts?\n"
            "A) The mass decreases because material flakes off\n"
            "B) The mass stays the same because no atoms are created or destroyed\n"
            "C) The mass increases because oxygen atoms bond to the iron\n"
            "D) The mass decreases because iron atoms are destroyed\n\n"
            "Answer with just the letter and a brief explanation."
        ),
        "category": "arc",
        "correct_answer": "C",
        "expected_contains": ["C", "oxygen"],
        "max_budget": 0.10,
    },
    {
        "id": "arc_biology_1",
        "prompt": (
            "Question: Which of the following is the correct order of biological organization "
            "from smallest to largest?\n"
            "A) Cell → Tissue → Organ → Organ system → Organism\n"
            "B) Tissue → Cell → Organ → Organism → Organ system\n"
            "C) Organ → Tissue → Cell → Organ system → Organism\n"
            "D) Organism → Organ system → Organ → Tissue → Cell\n\n"
            "Answer with just the letter and a brief explanation."
        ),
        "category": "arc",
        "correct_answer": "A",
        "expected_contains": ["A"],
        "max_budget": 0.10,
    },
    {
        "id": "arc_earthsci_1",
        "prompt": (
            "Question: What causes the phases of the Moon?\n"
            "A) Earth's shadow falling on the Moon\n"
            "B) The Moon's distance from Earth changing\n"
            "C) The relative positions of the Sun, Earth, and Moon as the Moon orbits Earth\n"
            "D) The Moon rotating on its axis\n\n"
            "Answer with just the letter and a brief explanation."
        ),
        "category": "arc",
        "correct_answer": "C",
        "expected_contains": ["C"],
        "max_budget": 0.10,
    },

    # ═══════════════════════════════════════════════════════════════
    # 5. MATH — Competition Level (GPT-4: 42.5%)
    # ═══════════════════════════════════════════════════════════════

    {
        "id": "math_algebra_1",
        "prompt": (
            "Find all real solutions to the equation: x^2 - 5x + 6 = 0. "
            "Show your complete step-by-step solution."
        ),
        "category": "math",
        "correct_answer": "x=2, x=3",
        "expected_contains": ["2", "3"],
        "max_budget": 0.30,
    },
    {
        "id": "math_combinatorics_1",
        "prompt": (
            "How many ways can you arrange the letters in the word 'MISSISSIPPI'? "
            "Show your complete step-by-step solution using the multinomial coefficient."
        ),
        "category": "math",
        "correct_answer": "34650",
        "expected_contains": ["34650"],
        "max_budget": 0.30,
    },
    {
        "id": "math_geometry_1",
        "prompt": (
            "A right triangle has legs of length 5 and 12. "
            "What is the area of the circle inscribed in this triangle? "
            "Express your answer in terms of pi. Show your complete step-by-step solution."
        ),
        "category": "math",
        "correct_answer": "4pi",
        "expected_contains": ["4"],
        "max_budget": 0.30,
    },
    {
        "id": "math_numbertheory_1",
        "prompt": (
            "What is the remainder when 2^100 is divided by 7? "
            "Show your complete step-by-step solution using modular arithmetic."
        ),
        "category": "math",
        "correct_answer": "2",
        "expected_contains": ["2"],
        "max_budget": 0.30,
    },
]


def get_gpt4_tasks_by_category(category: str):
    """Filter GPT-4 tasks by category."""
    return [t for t in GPT4_BENCHMARK_TASKS if t["category"] == category]


def get_categories():
    """Return list of unique categories."""
    return list(GPT4_REFERENCE_SCORES.keys())
