import streamlit as st
import ollama
from PIL import Image
import io

st.title("🏥 Gemma 4: Offline Medical Scribe")

# Uploading the chart
uploaded_file = st.file_uploader("Upload Medical Chart", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Chart Preview")
    
    if st.button("Analyze Offline"):
        with st.spinner("Gemma 4 is reading the chart..."):
            # Convert image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            
            # Send to local Gemma 4 model
            response = ollama.generate(
                model='gemma4:e4b',
                prompt="Extract patient vitals and medications from this chart.",
                images=[img_byte_arr.getvalue()]
            )
            st.success("Analysis Complete!")
            st.write(response['response'])
