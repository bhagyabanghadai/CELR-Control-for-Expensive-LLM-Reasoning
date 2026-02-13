import unittest
from unittest.mock import MagicMock
from celr.cortex.router import Router
from celr.core.llm import LLMUsage

class TestRouter(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock()
        self.router = Router(self.mock_llm)

    def test_classify_simple_greeting(self):
        # Mock LLM response for a simple query
        self.mock_llm.generate.return_value = (
            '```json\n{ "type": "DIRECT", "reason": "Simple greeting" }\n```',
            LLMUsage()
        )
        
        category, reason = self.router.classify("Hello, how are you?")
        self.assertEqual(category, "DIRECT")
        self.assertEqual(reason, "Simple greeting")

    def test_classify_complex_math(self):
        # Mock LLM response for a complex query
        self.mock_llm.generate.return_value = (
            '{ "type": "REASONING", "reason": "Requires calculation" }',
            LLMUsage()
        )
        
        category, reason = self.router.classify("Calculate the 100th Fibonacci number.")
        self.assertEqual(category, "REASONING")
        self.assertEqual(reason, "Requires calculation")

    def test_classify_fallback(self):
        # Simulate LLM failure (exception)
        self.mock_llm.generate.side_effect = Exception("API error")
        
        category, reason = self.router.classify("Something unclear")
        # Should default to DIRECT (Fast Path) as per design decision
        self.assertEqual(category, "DIRECT")
        self.assertIn("failed", reason)

if __name__ == '__main__':
    unittest.main()
