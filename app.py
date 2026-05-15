import os
import tempfile

import streamlit as st

from scribe import process_chart, run_agent

st.set_page_config(
    page_title="Gemma Medical Scribe",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 Gemma-Medical-Scribe")
st.subheader("Offline Clinical Reasoning at the Edge")
st.caption("Powered by Gemma 4 E4B · Running locally via Ollama · Zero internet required")

with st.sidebar:
    st.markdown("## System Status")
    st.success("Gemma 4 E4B — Online (Local)")
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
                    image_path = None
                    with tempfile.NamedTemporaryFile(
                        suffix=f"_{uploaded_chart.name}", delete=False
                    ) as tmp_file:
                        tmp_file.write(uploaded_chart.getvalue())
                        image_path = tmp_file.name
                    try:
                        result = process_chart(image_path)
                    finally:
                        if image_path and os.path.exists(image_path):
                            os.unlink(image_path)
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
