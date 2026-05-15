try:
    import ollama
except ImportError:  # pragma: no cover - used in test environments where ollama is not installed
    class _OllamaFallback:
        @staticmethod
        def chat(*_args, **_kwargs):
            raise RuntimeError("The 'ollama' package is required to run inference.")

    ollama = _OllamaFallback()

TERM_DEFINITIONS = {
    "dyspnea": "Dyspnea means shortness of breath or difficult breathing.",
}


def enrich_prompt(prompt):
    """Appends known medical definitions requested by the prompt."""
    enriched_prompt = prompt
    lowered = prompt.lower()
    for term, definition in TERM_DEFINITIONS.items():
        if term in lowered:
            enriched_prompt = f"{enriched_prompt}\n\nReference definition: {definition}"
    return enriched_prompt


def generate_medical_summary(transcript, model="gemma4:e4b"):
    """Summarizes medical content into a SOAP note."""
    system_prompt = (
        "You are a medical scribe. Summarize the provided medical content into a clear SOAP note "
        "(Subjective, Objective, Assessment, Plan). Keep it clinically accurate and concise."
    )
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript},
        ],
    )
    return response["message"]["content"]


def run_agent(prompt, model="gemma4:e4b"):
    """Main helper used for text prompts."""
    return generate_medical_summary(enrich_prompt(prompt), model=model)


def generate_medical_note_with_raw(transcript, model="gemma4:e4b"):
    """
    Summarizes a medical transcript into a structured SOAP note.
    Returns both the note and the raw response.
    """
    system_prompt = (
        "You are a medical scribe. Summarize the following doctor-patient conversation into a "
        "professional SOAP note with Subjective, Objective, Assessment, and Plan sections."
    )
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript},
        ],
    )
    return response["message"]["content"], response


def transcribe_chart(image_path):
    """Extracts text from a chart image with local OCR."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "OCR dependencies are missing. Install pillow and pytesseract."
        ) from exc

    img = Image.open(image_path)
    raw_text = pytesseract.image_to_string(img)
    return raw_text


def process_chart(image_path, model="gemma4:e4b"):
    """Runs OCR and returns a structured medical summary."""
    raw_notes = transcribe_chart(image_path)
    return generate_medical_summary(raw_notes, model=model)
