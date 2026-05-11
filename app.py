import streamlit as st
import ollama
from scribe import generate_medical_note

st.set_page_config(page_title="Gemma Medical Scribe", page_icon="⚕️")

st.title("⚕️ Gemma Offline Medical Scribe")
st.markdown("""
This tool uses **Gemma** via **Ollama** to transform medical transcripts into structured SOAP notes.
""")

with st.sidebar:
    st.header("Settings")
    model_name = st.text_input("Ollama Model", value="gemma")
    st.info("Ensure Ollama is running and the model is pulled.")

# Text area for the transcript
transcript_input = st.text_area(
    "Paste the medical transcript here:",
    height=300,
    placeholder="Doctor: How are you feeling today?\nPatient: I've had a headache for two days..."
)

if st.button("Generate SOAP Note"):
    if transcript_input.strip() == "":
        st.warning("Please enter a transcript first.")
    else:
        with st.spinner("Processing with Gemma..."):
            try:
                # We can wrap the generate_medical_note to use the custom model name if needed,
                # but for now we'll use the existing function which defaults to 'gemma'.
                note = generate_medical_note(transcript_input)
                
                st.subheader("Generated SOAP Note")
                st.markdown(note)
                
                st.download_button(
                    label="Download Note as TXT",
                    data=note,
                    file_name="soap_note.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Error connecting to Ollama: {e}")
                st.info("Check if Ollama is running (`ollama serve`) and the model is available.")

st.divider()
st.caption("Privacy focused: All processing happens locally on your machine.")
