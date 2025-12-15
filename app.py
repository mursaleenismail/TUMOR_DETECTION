
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
from io import BytesIO

model = load_model("model.keras")

st.set_page_config(
    page_title="Brain Tumor Detection AI",
    page_icon="🧠",
    layout="centered"
)

st.markdown(
    """
    <h1 style='text-align: center; color: #FF0000;'>Brain Tumor Detection System (AI Powered)</h1>
    <p style='text-align: center; color: #6A5ACD;'>Upload MRI images to detect brain tumor with confidence score.</p>
    """,
    unsafe_allow_html=True
)

st.write("---")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)
    
   
    image_array = np.array(image)
    image_array = cv2.resize(image_array, (224, 224))
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    st.write("---")
    st.markdown("<h3 style='color: #4B0082;'>Prediction Result</h3>", unsafe_allow_html=True)
    
   
    if st.button("Detect Tumor"):
        prediction = model.predict(image_array)
        confidence = prediction[0][0] * 100
   
        if prediction[0][0] > 0.5:
            st.error(f"🧠 Tumor Detected! (Confidence: {confidence:.2f}%)")
            result_text = f"Tumor Detected 🧠 (Confidence: {confidence:.2f}%)"
        else:
            st.success(f"✅ No Tumor Detected (Confidence: {100 - confidence:.2f}%)")
            result_text = f"No Tumor Detected ✅ (Confidence: {100 - confidence:.2f}%)"

        
        report_text = f"""
        Brain Tumor Detection Report
        ---------------------------
        Prediction Result: {result_text}
        """
        report_bytes = report_text.encode()
        
        # Add download button
        st.download_button(
            label=" Download Prediction Report",
            data=report_bytes,
            file_name="prediction_report.txt",
            mime="text/plain"
        )
