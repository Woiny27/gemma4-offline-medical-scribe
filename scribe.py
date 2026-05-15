import ollama
from typing import Callable, List


MEDICAL_TERMS_DICTIONARY = {
    "hypertension": "Persistently elevated arterial blood pressure.",
    "tachycardia": "A heart rate above the normal resting range, usually over 100 beats per minute in adults.",
    "bradycardia": "A slower than normal heart rate, typically below 60 beats per minute in adults.",
    "dyspnea": "Subjective sensation of difficult or uncomfortable breathing.",
    "edema": "Swelling caused by excess fluid in body tissues.",
    "pharyngitis": "Inflammation of the pharynx, commonly causing sore throat.",
    "otitis media": "Infection or inflammation of the middle ear.",
    "gastroenteritis": "Inflammation of the stomach and intestines, often causing vomiting and diarrhea.",
    "migraine": "A recurrent, often unilateral headache disorder associated with nausea and/or sensitivity to light and sound.",
    "diabetes mellitus": "A chronic metabolic condition characterized by elevated blood glucose levels.",
}


def medical_lookup(term: str) -> str:
    """Returns a short dictionary-style definition for a medical term."""
    normalized = term.strip().lower()
    if not normalized:
        return "No term provided."
    return MEDICAL_TERMS_DICTIONARY.get(normalized, "Definition not found in local medical dictionary.")


def _extract_terms(raw_terms_content: str) -> List[str]:
    if not raw_terms_content:
        return []
    lowered = raw_terms_content.strip().lower()
    if lowered in {"none", "n/a", "no terms"}:
        return []
    terms = [term.strip() for term in raw_terms_content.split(",")]
    return [term for term in terms if term]


def run_agent(
    prompt: str,
    model: str = "gemma4:e4b",
    lookup_func: Callable[[str], str] = medical_lookup,
):
    """
    Runs a 2-stage thinking-style medical agent:
    1) Extract terms to define.
    2) Lookup each term and generate a SOAP note with enriched context.
    """
    terms_response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a medical AI agent. Extract key medical terms from the prompt that need "
                    "definition. Return only a comma-separated list of terms."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    terms_content = terms_response["message"]["content"]
    terms = _extract_terms(terms_content)

    context_lines = []
    for term in terms:
        definition = lookup_func(term)
        context_lines.append(f"- {term}: {definition}")
    context = "\n".join(context_lines)

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
                    f"{prompt}\n\nUse these definitions as context:\n{context}\n\nGenerate a complete SOAP note."
                ),
            },
        ],
    )
    return final_response["message"]["content"], final_response

def generate_medical_note_with_raw(transcript, model='gemma'):
    """
    Summarizes a medical transcript into a structured SOAP note.
    Returns both the note and the raw response.
    """
    return run_agent(transcript, model=model)

if __name__ == "__main__":
    # Placeholder for actual transcription logic
    sample_transcript = "Patient: I've been having a persistent cough for 2 weeks. Doctor: Any fever?"
    print("Generating SOAP Note...")
    note, raw = generate_medical_note_with_raw(sample_transcript)
    print(note)
