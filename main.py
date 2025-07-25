# imports

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense
from keras.callbacks import EarlyStopping

import re
import streamlit as st

# load IMDB dataset word index
word_index = imdb.get_word_index()

# create index to word mapping
index_to_word = {
    index:word
    for word, index in word_index.items()
    }

# load trained RNN model
model = keras.models.load_model("simple_rnn_imdb.h5")

# Helper Functions to Enable Prediction
## Function 1: Decoding the moview review from encoded format to text
def decode_review(encoded_review):
    """
    Decode a review from encoded format to text.
    
    Args:
        encoded_review (list): A list of integers representing the encoded review.
    Returns:
        str: The decoded review as a string.
    """
    # Adjust the indices to match the IMDB dataset's word index
    return " ".join([index_to_word.get(i - 3, "?") for i in encoded_review])

# Function 2: Preprocessing the input text for prediction
def preprocess_text(text, maxlen=500):
    """
    Preprocess the input text for prediction.
    
    Args:
        text (str): The input text to preprocess.
        maxlen (int): The maximum length of the sequence.
    Returns:
        np.ndarray: The preprocessed text as a padded sequence.
    """
    # Encode the text
    words = text.lower()
    # remove punctuation
    words = re.sub(r"[^a-zA-Z0-9\s]", "", words)
    words = words.split()
    encoded_text = [word_index.get(word, 2) + 3 for word in words] # Adjusting for IMDB's word index
    # Pad the sequence to the maximum length
    padded_text = pad_sequences([encoded_text], maxlen=maxlen, padding="pre")

    return padded_text

# Function 3: Predicting the sentiment of a review
def predict_review_sentiment(review):
    """
    Predict the sentiment of a review.
    
    Args:
        encoded_review (list): A list of integers representing the encoded review.
    Returns:
        str: The predicted sentiment ("Positive" or "Negative").
    """
    # preprocess the review
    pre_processed_review = preprocess_text(review)
    
    # Make prediction
    prediction = model.predict(pre_processed_review)
    
    # prediction score
    score = prediction[0][0]

    # Return prediction and score
    return prediction, score


# Build streamlit app

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (Positive or Negative):")

# get user input
user_review = st.text_area("Movie Review", height=200)

# create a button to trigger prediction
if st.button("Predict Sentiment"):
    if user_review:
                
        # Make prediction
        prediction, score = predict_review_sentiment(user_review)

        # predicted sentiment
        sentiment = "Positive" if score > 0.5 else "Negative"

        # Display the result
        st.write(f"Predicted Sentiment: {sentiment}")
        st.write(f"Prediction Score: {score:.4f}")
    else:
        st.write("Please enter a movie review to predict its sentiment.")


        