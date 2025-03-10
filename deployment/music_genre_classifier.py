import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.image import resize



# Function

@st.cache_resource()

def load_model():

    model = tf.keras.models.load_model("Trained_model.h5")

    return model


# Load and preprocess audio data

def load_and_preprocess_data(file_path, target_shape=(150, 150)):

    data = []

    audio_data, sample_rate = librosa.load(file_path, sr=None)

    # Perform preprocessing (e.g., convert to Mel spectrogram and resize)

    chunk_duration = 4  # seconds

    overlap_duration = 2  # seconds



    # Convert durations to samples

    chunk_samples = chunk_duration * sample_rate

    overlap_samples = overlap_duration * sample_rate



    # Calculate the number of chunks

    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1



    # Iterate over each chunk

    for I in range(num_chunks):

        start = I * (chunk_samples - overlap_samples)

        end = start + chunk_samples



        chunk = audio_data[start:end]



        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)

        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)

        data.append(mel_spectrogram)



    return np.array(data)


# TensorFlow Model Prediction

def model_prediction(X_test):

    model = load_model()

    y_pred = model.predict(X_test)

    predicted_categories = np.argmax(y_pred, axis=1)

    unique_elements, counts = np.unique(predicted_categories, return_counts=True)

    max_count = np.max(counts)

    max_elements = unique_elements[counts == max_count]

    return max_elements[0]






# Placeholder function for genre prediction
def predict_genre(audio_file):
    X_test = load_and_preprocess_data(audio_file)
    result_index = model_prediction(X_test)
    st.balloons()
    label = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

    return label[result_index]

# Streamlit App
st.title("🎵 Music Genre Classifier 🎶")
st.write("Upload an audio file and click 'Predict Genre' to identify the music genre.")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    if st.button("Predict Genre"):
        try:
            # Predict the genre
            genre = predict_genre(uploaded_file)
            st.success(f"Predicted Genre: **{genre}**")
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
else:
    st.info("Please upload an audio file to begin.")
