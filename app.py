import streamlit as st
import ollama
from PIL import Image
import io
import os

st.set_page_config(page_title="Offline Rural Health Assistant", layout="wide", page_icon="🏥")

# --- Asset Management ---
# Ensure your assets/ folder exists and contains 'logo.png' for this to show
logo_path = "assets/logo.png"
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=150)

st.title("🏥 Gemma 4: Offline Medical Scribe")
st.info("Operating Mode: **100% Offline** | Model: `gemma4:e4b`")

# Sidebar for controls
with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload Medical Chart Scan", type=["jpg", "jpeg", "png"])
    temperature = st.slider("Model Temperature (Low = more precise)", 0.0, 1.0, 0.0, 0.1)
    st.divider()
    st.caption("Powered by local Gemma 4 weights. No data leaves this device.")

if uploaded_file:
    col1, col2 = st.columns(2)
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Medical Chart", use_container_width=True)
    
    with col2:
        if st.button("Analyze Chart Offline", type="primary"):
            with st.spinner("Gemma 4 is reasoning through the chart (this may take 10-30s)..."):
                try:
                    # Convert image to bytes for Ollama
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format='JPEG')
                    img_bytes = img_byte_arr.getvalue()

                    # Call the local Gemma 4 E4B model
                    response = ollama.generate(
                        model='gemma4:e4b',
                        prompt="Extract the following from this medical chart: 1. Patient Vitals, 2. Current Medications, 3. Recommended Follow-up. Format as a clean markdown list.",
                        images=[img_bytes],
                        options={'temperature': temperature}
                    )
                    
                    analysis = response['response']
                    
                    # --- Display Results ---
                    st.subheader("Extracted Clinical Data")
                    st.markdown(analysis)
                    
                    # Allow clinician to download the report
                    st.download_button(
                        label="💾 Download Clinical Note",
                        data=analysis,
                        file_name=f"scribe_note_{uploaded_file.name}.txt",
                        mime="text/plain"
                    )
                    st.success("Analysis complete.")

                except Exception as e:
                    st.error(f"Inference Error: {str(e)}")
                    st.warning("Check if Ollama is running (`ollama serve`) and model 'gemma4:e4b' is loaded.")
else:
    st.write("### Instructions")
    st.write("1. Upload a clear photo of the handwritten or printed medical chart.")
    st.write("2. Adjust temperature in the sidebar (0.0 is recommended for clinical accuracy).")
    st.write("3. Click 'Analyze Chart Offline' to begin.")
