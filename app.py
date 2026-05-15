import streamlit as st

from scribe import DEFAULT_MODEL, generate_medical_note_with_raw

st.set_page_config(page_title="Gemma Medical Scribe", page_icon="⚕️")

st.title("⚕️ Gemma Offline Medical Scribe")
st.write(
    "This tool uses **Gemma 4 E4B** via **Ollama** to transform medical transcripts into "
    "structured SOAP notes and can consult a built-in offline medical dictionary when needed."
)

with st.sidebar:
    st.header("Settings")
    model_name = st.text_input("Ollama Model", value=DEFAULT_MODEL)
    st.info("Ensure Ollama is running and the model is pulled locally.")

transcript_input = st.text_area(
    "Paste the medical transcript here:",
    height=300,
    placeholder="Doctor: How are you feeling today?\nPatient: I've had a headache for two days...",
)

if st.button("Generate SOAP Note"):
    if transcript_input.strip() == "":
        st.warning("Please enter a transcript first.")
    else:
        with st.spinner("Processing with Gemma..."):
            try:
                note, raw_response = generate_medical_note_with_raw(transcript_input, model=model_name)

                st.subheader("Generated SOAP Note")
                st.markdown(note)

                with st.expander("View Raw API Response"):
                    st.write("API Response:", raw_response)

                st.download_button(
                    label="Download Note as TXT",
                    data=note,
                    file_name="soap_note.txt",
                    mime="text/plain",
                )
            except Exception as error:
                st.error(f"Error connecting to Ollama: {error}")
                st.info("Check if Ollama is running (`ollama serve`) and the model is available.")

st.divider()
st.caption("Privacy focused: all processing happens locally on your machine.")
