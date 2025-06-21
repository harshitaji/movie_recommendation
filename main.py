import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
import numpy as np

# Initialize session state
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

# Load model and word index
word_index = imdb.get_word_index()
model = load_model('simple_rnn_imdb.h5')  # Make sure this file exists

def preprocess_text(text):
    words = text.lower().split()
    encoded = [word_index.get(word, 2) + 3 for word in words]
    padded = sequence.pad_sequences([encoded], maxlen=500)
    return padded

# UI Elements
st.title("🎬 IMDB Review Sentiment Analysis")
st.markdown("Enter a movie review to analyze its sentiment")

# Text area
user_input = st.text_area(
    'Movie Review:',
    value=st.session_state.user_input,
    height=150,
    key="review_text",
    placeholder="Type or paste a movie review here..."
)

# Add example buttons below the text area
col1, col2 = st.columns(2)
with col1:
    if st.button("😊 Positive Example"):
        st.session_state.user_input = "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout."
        st.rerun()
with col2:
    if st.button("😠 Negative Example"):
        st.session_state.user_input = "What a terrible film. The story made no sense and the acting was wooden at best."
        st.rerun()

# Prediction logic
if st.button('Analyze Sentiment', type="primary"):
    if st.session_state.user_input.strip():
        with st.spinner('Analyzing...'):
            processed = preprocess_text(st.session_state.user_input)
            prediction = model.predict(processed)
            confidence = prediction[0][0]
            
            if confidence > 0.5:
                st.success(f"😊 Positive sentiment (confidence: {confidence:.2%})")
            else:
                st.error(f"😠 Negative sentiment (confidence: {1-confidence:.2%})")
            
            st.progress(float(confidence) if confidence > 0.5 else float(1-confidence))
    else:
        st.warning("Please enter a review first!")