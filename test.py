import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import scribe


class ScribeTests(unittest.TestCase):
    def test_medical_lookup_returns_known_definition(self):
        self.assertIn("shortness of breath", scribe.medical_lookup("Dyspnea").lower())

    def test_generate_medical_note_returns_direct_response_without_tool_calls(self):
        mock_chat = Mock(return_value={"message": {"content": "SOAP note"}})

        with patch.object(scribe, "ollama", SimpleNamespace(chat=mock_chat)):
            note, raw_response = scribe.generate_medical_note_with_raw("Patient has a cough.", model="test-model")

        self.assertEqual(note, "SOAP note")
        self.assertEqual(raw_response["message"]["content"], "SOAP note")
        self.assertEqual(mock_chat.call_count, 1)
        self.assertEqual(mock_chat.call_args.kwargs["tools"], scribe.OLLAMA_TOOLS)

    def test_generate_medical_note_runs_local_medical_lookup_when_requested(self):
        first_response = {
            "message": {
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "medical_lookup",
                            "arguments": {"term": "dyspnea"},
                        }
                    }
                ],
            }
        }
        final_response = {
            "message": {
                "content": "SOAP note noting shortness of breath."
            }
        }
        mock_chat = Mock(side_effect=[first_response, final_response])

        with patch.object(scribe, "ollama", SimpleNamespace(chat=mock_chat)):
            note, raw_response = scribe.generate_medical_note_with_raw(
                "Patient reports dyspnea while climbing stairs.",
                model="test-model",
            )

        self.assertEqual(note, "SOAP note noting shortness of breath.")
        self.assertEqual(raw_response, final_response)
        self.assertEqual(mock_chat.call_count, 2)

        second_call_messages = mock_chat.call_args_list[1].kwargs["messages"]
        tool_message = second_call_messages[-1]
        self.assertEqual(tool_message["role"], "tool")
        self.assertEqual(tool_message["name"], "medical_lookup")
        self.assertIn("shortness of breath", tool_message["content"].lower())


if __name__ == "__main__":
    unittest.main()
