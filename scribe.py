import re
try:
    import ollama
except ImportError:  # pragma: no cover - runtime dependency fallback
    class _OllamaFallback:
        @staticmethod
        def chat(*args, **kwargs):
            raise ImportError("ollama is required to generate notes.")

    ollama = _OllamaFallback()

MEDICAL_DB = {
    "dyspnea": "Difficult or labored breathing; shortness of breath.",
    "tachycardia": "Abnormally rapid heart rate, typically over 100 bpm.",
    "hypertension": "Persistently elevated blood pressure in the arteries.",
    "bradycardia": "Abnormally slow heart rate, typically below 60 bpm.",
    "edema": "Swelling caused by excess fluid trapped in body tissues.",
}


def medical_lookup(term):
    """
    Simulates an offline medical dictionary lookup.
    """
    return MEDICAL_DB.get(
        term.lower(),
        f"Definition for {term}: [Retrieved from local medical database]",
    )


tools = {
    "medical_lookup": medical_lookup,
}

def generate_medical_note_with_raw(transcript, model='gemma'):
    """
    Summarizes a medical transcript into a structured SOAP note.
    Returns both the note and the raw response.
    """
    system_prompt = (
        "You are a medical scribe. Your task is to summarize the following doctor-patient "
        "conversation into a professional, structured SOAP note (Subjective, Objective, "
        "Assessment, Plan). Ensure medical accuracy and a professional tone."
    )
    
    response = ollama.chat(model=model, messages=[
        {
            'role': 'system',
            'content': system_prompt,
        },
        {
            'role': 'user',
            'content': transcript,
        },
    ])
    return response['message']['content'], response


def _extract_terms(prompt):
    extracted_terms = []
    seen_terms = set()

    quoted_terms = re.findall(r"'([^']+)'", prompt)
    for term in quoted_terms:
        normalized = term.strip()
        lowered = normalized.lower()
        if normalized and lowered not in seen_terms:
            extracted_terms.append(normalized)
            seen_terms.add(lowered)

    lowered_prompt = prompt.lower()
    for known_term in MEDICAL_DB:
        if known_term in lowered_prompt and known_term not in seen_terms:
            extracted_terms.append(known_term)
            seen_terms.add(known_term)

    return extracted_terms


def run_agent(prompt, model='gemma'):
    """
    Runs an agentic SOAP-note generation flow with medical term enrichment.
    """
    terms = _extract_terms(prompt)
    definitions = [
        f"- {term}: {medical_lookup(term)}"
        for term in terms
    ]

    if definitions:
        enriched_prompt = (
            f"{prompt}\n\n"
            "Medical definitions to include in the summary:\n"
            f"{'\n'.join(definitions)}"
        )
    else:
        enriched_prompt = prompt

    note, _ = generate_medical_note_with_raw(enriched_prompt, model=model)
    return note

if __name__ == "__main__":
    prompt = "Analyze the term 'Dyspnea' and include its definition in the patient's summary."
    result = run_agent(prompt)
    print("Generated SOAP note successfully.")
