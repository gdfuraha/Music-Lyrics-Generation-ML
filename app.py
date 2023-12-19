import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle

# Function to load the model
def load_lyrics_model(model_path, tokenizer_path):
    try:
        model = load_model(model_path)
        with open(tokenizer_path, 'rb') as tokenizer_file:
            tokenizer = pickle.load(tokenizer_file)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        return None, None

# Function to generate lyrics
def generate_lyrics(seed_text, next_words, model, tokenizer, temperature=1.0):
    try:
        for _ in range(next_words):
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], padding='pre')
            predicted_probs = model.predict(token_list)[0]
            predicted_probs = predicted_probs / temperature
            predicted_id = np.random.choice(len(predicted_probs), p=predicted_probs)
            output_word = tokenizer.index_word.get(predicted_id, "")
            seed_text += " " + output_word
        return seed_text
    except Exception as e:
        st.error(f"Error generating lyrics: {str(e)}")
        return None

# Streamlit UI
st.title("Music Lyrics Generator")

# Load the model and tokenizer
model, tokenizer = load_lyrics_model('model2.h5', 'tokenizer.pkl')  

# User input for seed text
seed_text = st.text_area("Enter seed text:")

# Generate button and check if seed text is provided
if st.button("Generate Lyrics") and seed_text:
    # Generate lyrics
    generated_text = generate_lyrics(seed_text, next_words=50, model=model, tokenizer=tokenizer)

    # Display generated lyrics
    if generated_text:
        st.subheader("Generated Lyrics:")
        st.write(generated_text)
else:
    st.warning("Please enter seed text before generating lyrics.")
