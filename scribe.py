import json

try:
    import ollama
except ImportError:
    ollama = None

DEFAULT_MODEL = "gemma4:e4b"


def medical_lookup(term):
    medical_db = {
        "dyspnea": "Difficult or labored breathing; shortness of breath.",
        "tachycardia": "Abnormally rapid heart rate, typically over 100 bpm.",
        "hypertension": "Persistently elevated blood pressure in the arteries.",
        "bradycardia": "Abnormally slow heart rate, typically below 60 bpm.",
        "edema": "Swelling caused by excess fluid trapped in body tissues.",
    }
    return medical_db.get(term.lower(), f"Definition for {term}: [Retrieved from local medical database]")


tools = {
    "medical_lookup": medical_lookup,
}

OLLAMA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "medical_lookup",
            "description": "Look up medical terms in the local offline medical dictionary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "term": {
                        "type": "string",
                        "description": "The medical term that needs a definition.",
                    }
                },
                "required": ["term"],
            },
        },
    }
]


def _parse_tool_arguments(arguments):
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        if not arguments.strip():
            return {}
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return {"term": arguments}
    raise TypeError(f"Unsupported tool argument type: {type(arguments)!r}")


def _extract_tool_calls(response):
    message = response.get("message", {}) if isinstance(response, dict) else {}
    tool_calls = message.get("tool_calls") or []
    extracted_calls = []

    for tool_call in tool_calls:
        function_call = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
        tool_name = function_call.get("name")
        if not tool_name:
            continue
        extracted_calls.append(
            {
                "name": tool_name,
                "arguments": _parse_tool_arguments(function_call.get("arguments")),
            }
        )

    return extracted_calls


def _run_tool_calls(messages, tool_calls):
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool = tools.get(tool_name)
        if tool is None:
            continue

        tool_result = tool(**tool_call["arguments"])
        messages.append(
            {
                "role": "tool",
                "name": tool_name,
                "content": tool_result,
            }
        )

    return messages


def generate_medical_note_with_raw(transcript, model=DEFAULT_MODEL):
    """
    Summarizes a medical transcript into a structured SOAP note.
    Returns both the note and the raw response.
    """
    if ollama is None:
        raise RuntimeError("The ollama package is required. Install dependencies with `pip install -r requirements.txt`.")

    system_prompt = (
        "You are a medical scribe. Summarize the doctor-patient conversation into a professional, "
        "structured SOAP note (Subjective, Objective, Assessment, Plan). Use the local "
        "`medical_lookup` tool when a medical term needs clarification before finalizing the note."
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": transcript,
        },
    ]

    response = ollama.chat(model=model, messages=messages, tools=OLLAMA_TOOLS)
    tool_calls = _extract_tool_calls(response)

    if tool_calls:
        assistant_message = response.get("message")
        if assistant_message:
            messages.append(assistant_message)
        messages = _run_tool_calls(messages, tool_calls)
        response = ollama.chat(model=model, messages=messages, tools=OLLAMA_TOOLS)

    return response["message"]["content"], response


if __name__ == "__main__":
    sample_transcript = "Patient reports dyspnea and ankle edema for the past week."
    print("Generating SOAP Note...")
    note, raw = generate_medical_note_with_raw(sample_transcript)
    print(note)
