import ollama
import pytesseract
import streamlit as st
from PIL import Image


def generate_medical_summary(patient_notes: str) -> str:
    try:
        response = ollama.chat(
            model="gemma4:e4b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional medical scribe. "
                        "Structure notes clearly and flag any urgent findings."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Summarize these notes into a SOAP format: {patient_notes}",
                },
            ],
        )
        return response["message"]["content"]
    except Exception as exc:
        raise RuntimeError(
            "Unable to reach local Ollama model. Ensure `ollama serve` is running "
            "and `gemma4:e4b` is available."
        ) from exc


st.set_page_config(page_title="Gemma Offline Medical Scribe", page_icon="⚕️")
st.title("⚕️ Gemma Offline Medical Scribe")
st.caption("Privacy focused: all processing runs locally.")

tab1, tab2 = st.tabs(["Manual Notes", " Scan Chart"])

with tab1:
    notes = st.text_area(
        "Enter raw patient observations:",
        height=200,
        placeholder=(
            "e.g. 34F, persistent cough 3 weeks, night sweats, "
            "weight loss 4kg, T 38.4°C, SpO2 94%…"
        ),
    )
    if st.button("Generate SOAP Note", type="primary"):
        if notes.strip():
            try:
                with st.spinner("Gemma 4 E4B reasoning locally…"):
                    summary = generate_medical_summary(notes)
                st.markdown("### Generated SOAP Note")
                st.markdown(summary)
                st.download_button("Download Note", summary, file_name="soap_note.txt")
            except RuntimeError as err:
                st.error(str(err))
        else:
            st.warning("Please enter patient observations first.")

with tab2:
    uploaded = st.file_uploader(
        "Upload a photo of the patient chart",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded Chart", use_container_width=True)
        if st.button("Transcribe & Analyse", type="primary"):
            with st.spinner("Running OCR + Gemma 4 E4B locally…"):
                raw_text = pytesseract.image_to_string(img)
            st.markdown("### OCR Extracted Text")
            st.code(raw_text)

            if not raw_text.strip():
                st.warning(
                    "No readable text was found in the uploaded image. "
                    "Please upload a clearer chart image."
                )
            else:
                try:
                    with st.spinner("Generating SOAP note from OCR text…"):
                        summary = generate_medical_summary(raw_text)
                    st.markdown("### Generated SOAP Note")
                    st.markdown(summary)
                    st.download_button("Download Note", summary, file_name="soap_note.txt")
                except RuntimeError as err:
                    st.error(str(err))
