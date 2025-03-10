** Music Genre Classification  **

**Project Overview  **
This project aims to classify music genres using machine learning. Audio files are processed into Mel spectrograms, which serve as input to a Convolutional Neural Network (CNN). The model extracts spatial patterns from these spectrograms and predicts the genre of a given audio file. A Streamlit web application is also built to provide an interactive interface for users to upload and classify their audio files.  

**Dataset Preparation ** 
Audio files are processed by loading them using the librosa library. Each file is split into overlapping chunks to generate more training samples. Mel spectrograms are computed for each chunk to capture essential frequency-based features. The spectrograms are resized to a uniform shape for consistent model input. Labels are assigned to each chunk and converted into categorical format. The dataset is then split into training and testing sets in an 80:20 ratio.  

** Model Architecture**  
The Convolutional Neural Network (CNN) consists of multiple layers to extract and learn patterns from Mel spectrograms. Convolutional layers capture frequency and time-based features, while max pooling layers reduce dimensionality. Dropout layers prevent overfitting by randomly deactivating neurons during training. Extracted features are flattened and passed through fully connected layers to learn complex relationships. A softmax output layer generates probability scores for each genre category. The model is compiled using the Adam optimizer with a learning rate of 0.0001 and categorical cross-entropy as the loss function. Training is performed for 30 epochs with a batch size of 32.  
**
Model Evaluation ** 
The trained model is evaluated using loss and accuracy metrics for both training and testing datasets. A confusion matrix visualizes classification errors, while a classification report provides precision, recall, and F1-score for each genre. These metrics help assess the model’s overall performance and predictive capabilities.  

**Real-World Testing ** 
To test the model in real-world scenarios, an independent function is used to predict the genre of an uploaded audio file. The file is loaded and split into chunks, with each chunk converted into a Mel spectrogram. The trained model processes these spectrograms, and the most frequently predicted genre is chosen as the final output.  

**Streamlit Web Application  **
A Streamlit-based web application provides an interactive interface for users. The application allows users to upload audio files in .mp3 or .wav format. The model processes the uploaded file and displays the predicted genre along with confidence scores. A loading indicator ensures a smooth user experience, and visualizations help users understand how the classification is performed.   
