from scribe import generate_medical_note

def test_scribe():
    print("Testing Gemma Offline Medical Scribe...")
    
    # A more detailed test transcript
    test_transcript = """
    Doctor: Good morning, Mr. Smith. What brings you in today?
    Patient: I've had this sharp pain in my lower back for about three days now. It's worse when I try to bend over.
    Doctor: I see. Did you injure it recently?
    Patient: I was lifting some heavy boxes in the garage on Saturday.
    Doctor: Any numbness or tingling in your legs?
    Patient: No, just the sharp pain in the back.
    Doctor: Alright. I'm going to examine your back. Please stand up.
    """
    
    try:
        print("\nSending transcript to Ollama (Gemma)...")
        note = generate_medical_note(test_transcript)
        
        print("\n--- Generated SOAP Note ---")
        print(note)
        print("----------------------------")
        
        if "Subjective" in note or "Assessment" in note:
            print("\nSuccess: SOAP structure detected in the response.")
        else:
            print("\nWarning: Response received, but standard SOAP headers might be missing.")
            
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure Ollama is running and the 'gemma' model is pulled (ollama pull gemma).")

if __name__ == "__main__":
    test_scribe()
