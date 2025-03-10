# Music Genre Classification For Predictive Analysis Using Deep Learning 🎶🎧

## Overview
This project is a **Music Genre Classification For Predictive Analysis Using Deep Learning** built using **TensorFlow, Streamlit, and Librosa**. The system allows users to upload an audio file, process it, and predict the genre using a deep learning model.

## Deployment Link
See the demo here: [**Music Genre Classification App**](https://musicgenreclassification-hftslkegfum3fjozyhcaem.streamlit.app/)

## Features
- **Upload an Audio File:** Users can upload `.mp3` or `.wav` format files.
- **Preprocessing:** Audio is converted into Mel spectrograms for effective feature extraction.
- **Deep Learning Model**  A Convolutional Neural Network (CNN) is trained to identify patterns in audio features that correlate with different music genres.
- **Confidence Score:** The model provides a confidence score for classified genre prediction.
- **Interactive UI:** Built using Streamlit for a user-friendly experience.

## Dataset 
The dataset used for training the model is the **GTZAN Genre Collection**, sourced from Kaggle: [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification). 
For this project, only the **genres_original** folder was used, which contains 10 genres of audio files.
- Blues
- Classical
- Country
- Disco
- Hip-hop
- Jazz
- Metal
- Pop
- Reggae
- Rock

## Tech Stack 🛠️
- **Python** 🐍
- **TensorFlow** for deep learning
- **Streamlit** for web UI
- **Librosa** for audio processing
- **NumPy & Matplotlib** for data processing & visualization


## Usage 🎼
1. Open the **Streamlit web app**.
2. **Upload an audio file** (`.mp3`,`.wav`).
3. Click **Play Audio** to listen.
4. Click **Predict** to classify the genre.
5. View the **predicted music genre**!

## Example Output 📊
- **Uploaded File:** `song.mp3`
- **Predicted Genre:** 🎸 `Rock`
