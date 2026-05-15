import os
import tempfile

import streamlit as st

from scribe import process_chart, run_agent


def check_model_status():
    try:
        import ollama

        models = ollama.list()
        available = any(
            model.get("model") == "gemma4:e4b"
            for model in models.get("models", [])
            if isinstance(model, dict)
        )
        return True, available
    except Exception:
        return False, False


st.set_page_config(
    page_title="Gemma Medical Scribe",
    page_icon="⚕️",
    layout="wide",
)

st.title("⚕️ Gemma Medical Scribe")
st.subheader("Offline Clinical Reasoning at the Edge")
st.caption("Powered by Gemma 4 E4B · Running locally via Ollama · Zero internet required")

with st.sidebar:
    ollama_online, model_available = check_model_status()
    st.markdown("## System Status")
    if ollama_online and model_available:
        st.success("Gemma 4 E4B — Online (Local)")
    else:
        st.error("Gemma 4 E4B — Offline/Unavailable")
    st.error("Internet — Disconnected")
    st.info("Mode: Full Offline Edge")
    st.markdown("---")
    st.markdown("**Model:** gemma4:e4b")
    st.markdown("**Runtime:** Ollama")
    st.markdown("**OCR Engine:** Tesseract")

st.markdown("### Text Prompt")
prompt = st.text_area(
    "Enter patient transcript or task",
    height=220,
    placeholder="Analyze the term 'Dyspnea' and include its definition in the patient's summary.",
)

st.markdown("### OCR Chart Transcription (Optional)")
uploaded_chart = st.file_uploader(
    "Upload scanned/photographed patient chart",
    type=["png", "jpg", "jpeg", "tiff", "bmp"],
)

if st.button("Generate Summary"):
    if not prompt.strip() and not uploaded_chart:
        st.warning("Please enter a prompt or upload a chart image.")
    else:
        with st.spinner("Processing locally with Gemma..."):
            try:
                if uploaded_chart:
                    allowed_suffixes = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
                    file_extension = os.path.splitext(uploaded_chart.name)[1].lower()
                    safe_suffix = file_extension if file_extension in allowed_suffixes else ".png"
                    with tempfile.TemporaryDirectory() as temp_dir:
                        image_path = os.path.join(temp_dir, f"chart{safe_suffix}")
                        with open(image_path, "wb") as temp_chart:
                            temp_chart.write(uploaded_chart.getvalue())
                        result = process_chart(image_path)
                else:
                    result = run_agent(prompt)

                st.subheader("Final SOAP Note")
                st.markdown(result)
                st.download_button(
                    label="Download Note as TXT",
                    data=result,
                    file_name="soap_note.txt",
                    mime="text/plain",
                )
            except Exception as exc:
                st.error(f"Failed to process input: {exc}")

st.divider()
st.caption("Privacy-focused: all processing happens locally on your machine.")
