import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb 
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

word_index = imdb.get_word_index()

reverse_word_index = {value : key for key , value in word_index.items()}

model = load_model('simple_rnn_imdb.h5')

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])


def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

##prediction

def predit_sentiment(review):
    preprocessed_inp = preprocess_text(review)
    if preprocessed_inp is None:
        raise ValueError("Preprocessing returned None!")

    prediction = model.predict(preprocessed_inp)

    sentiment = 'Positive(+)' if prediction[0][0] > 0.5 else 'Negative (-)'

    return sentiment , prediction[0][0]


import streamlit as st

st.title("IMDB Review Sentiment Analysis")

st.write("Enter a movie review to classify as positive(+) or negative(-)")

user_inp = st.text_area('Movie Review')

if st.button('Clasiify'):
     
    sentiment, score = predit_sentiment(user_inp)
    st.write(f"Sentiment: {sentiment}, Score: {score:.4f}")
