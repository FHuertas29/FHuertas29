import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load model
model = load_model('pneumonia_model.h5')

# Streamlit application
st.title('Pneumonia Prediction App')



with st.expander("ℹ️ - About this app", expanded=True):
    st.write("""
    Welcome to the Pneumonia Prediction App! This application utilizes a deep learning model to analyze chest X-ray images 
    and predict the likelihood of pneumonia. Our goal is to provide a quick and preliminary analysis to aid healthcare professionals.
    - **How to use:** Upload a chest X-ray image in JPEG format, and the model will analyze the image and provide a prediction.
    - **Note:** This app should not be used as a sole diagnostic tool. Consult a healthcare professional for an accurate diagnosis.
    """)

# Upload image through streamlit   
uploaded_file = st.file_uploader("Please upload an X-ray image of you chest", type="jpeg")

# Functionality of the app
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded X-ray image.', use_column_width=True)
    
    # Preprocess the image to match model input
    img = img.resize((128, 128))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.

    # Create a placeholder for the "Classifying..." message
    message_placeholder = st.empty()
    message_placeholder.text("Classifying...")
    # Preprocess and classify the image
    # Clear the "Classifying..." message
    message_placeholder.empty()

    # Make prediction
    prediction = model.predict(img)
    prediction_value = prediction[0][0] * 100  # Convert to percentage
    st.write('Result:')

    if prediction_value < 50:
        st.success(f"Good news! The X-ray image is classified as Normal. The model is {100 - prediction_value:.2f}% confident.")
    else:
        st.error(f"Bad news. The X-ray image is classified as Pneumonia. The model is {prediction_value:.2f}% confident.")


