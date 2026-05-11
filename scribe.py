import ollama
import json

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

if __name__ == "__main__":
    # Placeholder for actual transcription logic
    sample_transcript = "Patient: I've been having a persistent cough for 2 weeks. Doctor: Any fever?"
    print("Generating SOAP Note...")
    note, raw = generate_medical_note_with_raw(sample_transcript)
    print(note)
