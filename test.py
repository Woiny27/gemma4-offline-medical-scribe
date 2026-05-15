import unittest
from unittest.mock import patch

import scribe


class ScribeAgentTests(unittest.TestCase):
    def test_extract_terms_handles_empty_and_none(self):
        self.assertEqual(scribe._extract_terms(""), [])
        self.assertEqual(scribe._extract_terms("none"), [])
        self.assertEqual(scribe._extract_terms("No Terms"), [])

    def test_extract_terms_parses_csv(self):
        self.assertEqual(
            scribe._extract_terms("tachycardia, hypertension , edema"),
            ["tachycardia", "hypertension", "edema"],
        )

    @patch("scribe.ollama.chat")
    def test_run_agent_uses_lookup_context_and_returns_final_response(self, mock_chat):
        prompt = "Patient has tachycardia and edema."
        mock_chat.side_effect = [
            {"message": {"content": "tachycardia, edema"}},
            {"message": {"content": "SOAP NOTE"}},
        ]

        looked_up_terms = []

        def fake_lookup(term):
            looked_up_terms.append(term)
            return f"Definition for {term}"

        note, raw = scribe.run_agent(prompt, model="gemma4:e4b", lookup_func=fake_lookup)

        self.assertEqual(note, "SOAP NOTE")
        self.assertEqual(raw, {"message": {"content": "SOAP NOTE"}})
        self.assertEqual(looked_up_terms, ["tachycardia", "edema"])
        self.assertEqual(mock_chat.call_count, 2)

        second_call_messages = mock_chat.call_args_list[1].kwargs["messages"]
        self.assertIn("- tachycardia: Definition for tachycardia", second_call_messages[1]["content"])
        self.assertIn("- edema: Definition for edema", second_call_messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
