import unittest
from unittest.mock import patch

import scribe


class TestScribe(unittest.TestCase):
    def test_enrich_prompt_includes_dyspnea_definition(self):
        prompt = "Analyze the term Dyspnea and include it in the summary."
        enriched = scribe.enrich_prompt(prompt)
        self.assertIn("shortness of breath", enriched)

    @patch("scribe.ollama.chat")
    def test_run_agent_uses_enriched_prompt(self, mock_chat):
        mock_chat.return_value = {"message": {"content": "SOAP NOTE"}}
        result = scribe.run_agent("Analyze Dyspnea for this patient.")

        self.assertEqual(result, "SOAP NOTE")
        sent_prompt = mock_chat.call_args.kwargs["messages"][1]["content"]
        self.assertIn("Reference definition", sent_prompt)

    @patch("scribe.ollama.chat")
    def test_generate_medical_note_with_raw_returns_note_and_raw(self, mock_chat):
        mock_chat.return_value = {"message": {"content": "SOAP NOTE"}, "meta": "raw"}
        note, raw = scribe.generate_medical_note_with_raw("Transcript text")

        self.assertEqual(note, "SOAP NOTE")
        self.assertEqual(raw["meta"], "raw")

    @patch("scribe.generate_medical_summary")
    @patch("scribe.transcribe_chart")
    def test_process_chart_uses_ocr_then_summary(self, mock_ocr, mock_summary):
        mock_ocr.return_value = "raw chart"
        mock_summary.return_value = "structured"

        result = scribe.process_chart("/tmp/chart.png")

        self.assertEqual(result, "structured")
        mock_ocr.assert_called_once_with("/tmp/chart.png")
        mock_summary.assert_called_once_with("raw chart", model="gemma4:e4b")


if __name__ == "__main__":
    unittest.main()
