try:
    import ollama
except ModuleNotFoundError:  # pragma: no cover - used only when dependency is absent
    class _OllamaFallback:
        @staticmethod
        def chat(*args, **kwargs):
            raise ModuleNotFoundError("The 'ollama' package is required to run this module.")

    ollama = _OllamaFallback()


def _parse_terms(raw_terms):
    if not raw_terms:
        return []

    normalized = raw_terms.replace("\n", ",").replace(";", ",")
    terms = [term.strip() for term in normalized.split(",")]
    return [term for term in terms if term]


def extract_medical_terms(prompt, model="gemma4:e4b"):
    terms_response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a medical AI agent. Extract key medical terms from the prompt "
                    "that need definition. Return only a comma-separated list of terms."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    return _parse_terms(terms_response["message"]["content"])


def medical_lookup(term, model="gemma4:e4b"):
    definition_response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a medical dictionary. Provide a concise clinical definition for "
                    "the given term."
                ),
            },
            {"role": "user", "content": term},
        ],
    )
    return definition_response["message"]["content"].strip()


def _run_agent_with_raw(prompt, model="gemma4:e4b"):
    terms = extract_medical_terms(prompt, model=model)

    context = ""
    for term in terms:
        definition = medical_lookup(term, model=model)
        context += f"\n- {term}: {definition}"

    final_response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a professional medical scribe with access to a medical dictionary.",
            },
            {
                "role": "user",
                "content": (
                    f"{prompt}\n\nUse these definitions as context:{context}\n\n"
                    "Generate a complete SOAP note."
                ),
            },
        ],
    )
    return final_response["message"]["content"], final_response


def run_agent(prompt, model="gemma4:e4b"):
    note, _ = _run_agent_with_raw(prompt, model=model)
    return note


def generate_medical_note_with_raw(transcript, model="gemma4:e4b"):
    """
    Summarizes a medical transcript into a structured SOAP note.
    Returns both the note and the raw response.
    """
    return _run_agent_with_raw(transcript, model=model)

if __name__ == "__main__":
    # Placeholder for actual transcription logic
    sample_transcript = "Patient: I've been having a persistent cough for 2 weeks. Doctor: Any fever?"
    print("Generating SOAP Note...")
    note, raw = generate_medical_note_with_raw(sample_transcript)
    print(note)
