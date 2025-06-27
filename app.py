import streamlit as st
import librosa
import numpy as np
import pickle
import soundfile as sf
import tempfile
import pandas as pd

# Set Streamlit page configuration
st.set_page_config(page_title="Baby Cry Audio Prediction", layout="centered")

# App Title
st.title("👶 Baby Cry Audio Prediction")
st.markdown("Upload an audio file ( `.wav` ) of an infant and get the prediction.")

# Load the trained model
@st.cache_resource
def load_model():
    with open("audio_prediction_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Category labels (must match the order of model training labels)
categories = ["belly_pain", "burping", "discomfort", "hungry", "tired"]

# Feature extraction function (matching model input)
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=27)  # 🔧 Match model’s expected 27 features
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed.reshape(1, -1)

# File uploader
uploaded_file = st.file_uploader("Choose a `.wav` file", type=["wav"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    # Audio playback
    st.audio(uploaded_file, format='audio/wav')

    # Predict button
    if st.button("🔍 Predict Cry Type"):
        try:
            # Feature extraction and prediction
            features = extract_features(temp_path)
            prediction_index = model.predict(features)[0]
            proba = model.predict_proba(features)[0]

            # Get predicted label
            prediction_label = categories[prediction_index]

            # Show result
            st.success(f"🎧 Prediction: **{prediction_label}**")

            # Show probabilities in bar chart
            proba_df = pd.DataFrame({
                "Cry Type": categories,
                "Probability": proba
            })
            st.markdown("### 🔢 Prediction Probabilities")
            st.bar_chart(proba_df.set_index("Cry Type"))

        except Exception as e:
            st.error(f"❌ Error processing audio: {e}")
