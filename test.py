import unittest
from unittest.mock import patch

import scribe


class TestScribeAgentLoop(unittest.TestCase):
    @patch("scribe.ollama.chat")
    def test_extract_medical_terms_parses_list(self, mock_chat):
        mock_chat.return_value = {"message": {"content": "hypertension, diabetes , COPD"}}

        terms = scribe.extract_medical_terms("Patient has chronic conditions.")

        self.assertEqual(terms, ["hypertension", "diabetes", "COPD"])
        self.assertEqual(mock_chat.call_count, 1)

    @patch("scribe.ollama.chat")
    def test_run_agent_uses_lookup_context_for_final_note(self, mock_chat):
        mock_chat.side_effect = [
            {"message": {"content": "hypertension, diabetes"}},
            {"message": {"content": "High blood pressure."}},
            {"message": {"content": "A metabolic disease with elevated blood glucose."}},
            {"message": {"content": "Subjective: ...\nObjective: ...\nAssessment: ...\nPlan: ..."}},
        ]

        note = scribe.run_agent("Patient has hypertension and diabetes.")

        self.assertIn("Subjective:", note)
        self.assertEqual(mock_chat.call_count, 4)

        final_call_kwargs = mock_chat.call_args_list[-1].kwargs
        final_user_message = final_call_kwargs["messages"][-1]["content"]
        self.assertIn("- hypertension: High blood pressure.", final_user_message)
        self.assertIn(
            "- diabetes: A metabolic disease with elevated blood glucose.",
            final_user_message,
        )

    @patch("scribe.ollama.chat")
    def test_generate_medical_note_with_raw_returns_note_and_raw_response(self, mock_chat):
        mock_chat.side_effect = [
            {"message": {"content": "asthma"}},
            {"message": {"content": "A chronic inflammatory airway disease."}},
            {"message": {"content": "Final SOAP note"}},
        ]

        note, raw = scribe.generate_medical_note_with_raw("Patient has asthma.")

        self.assertEqual(note, "Final SOAP note")
        self.assertEqual(raw, {"message": {"content": "Final SOAP note"}})


if __name__ == "__main__":
    unittest.main()
