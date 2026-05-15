import unittest
import types
import sys
from unittest.mock import patch

sys.modules.setdefault("ollama", types.SimpleNamespace(chat=lambda **kwargs: {}))

import scribe


class TestScribeAgenticSkills(unittest.TestCase):
    def test_medical_lookup_known_term(self):
        self.assertEqual(
            scribe.medical_lookup("dyspnea"),
            "Difficult or labored breathing; shortness of breath.",
        )

    def test_medical_lookup_unknown_term(self):
        self.assertEqual(
            scribe.medical_lookup("foobar"),
            "Definition for foobar: [Retrieved from local medical database]",
        )

    @patch("scribe.ollama.chat")
    def test_run_agent_enriches_prompt_with_definition(self, mock_chat):
        mock_chat.return_value = {
            "message": {"content": "Structured SOAP note"},
        }

        result = scribe.run_agent(
            "Analyze the term 'Dyspnea' and include its definition in the patient's summary."
        )

        self.assertEqual(result, "Structured SOAP note")
        self.assertEqual(mock_chat.call_count, 1)
        call_kwargs = mock_chat.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "gemma")
        user_prompt = call_kwargs["messages"][1]["content"]
        self.assertIn(
            "Dyspnea: Difficult or labored breathing; shortness of breath.",
            user_prompt,
        )

    @patch("scribe.ollama.chat")
    def test_run_agent_without_known_term_uses_fallback(self, mock_chat):
        mock_chat.return_value = {
            "message": {"content": "Structured SOAP note"},
        }

        scribe.run_agent("Explain the term 'Xyzosis' in the assessment.")
        call_kwargs = mock_chat.call_args.kwargs
        user_prompt = call_kwargs["messages"][1]["content"]
        self.assertIn(
            "Xyzosis: Definition for Xyzosis: [Retrieved from local medical database]",
            user_prompt,
        )


if __name__ == "__main__":
    unittest.main()
