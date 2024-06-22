import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
from keras.utils import to_categorical
import os

def extract_melspectrogram(audio_path, n_mels=80, hop_length=512, n_fft=2048):
    #load the audio file
    scale, sr = librosa.load(audio_path, sr=None)

    #compute the mel-spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=scale, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)

    #convert to log scale (dB)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

    plt.figure(figsize=(25, 10))
    librosa.display.specshow(log_mel_spectrogram, 
                             x_axis="time",
                             y_axis="mel", 
                             sr=sr)
    plt.colorbar(format="%+2.f dB")
    plt.title("Log Mel Spectrogram")
    plt.show()

    return log_mel_spectrogram

def compute_binary_representation(log_mel_spectrogram, threshold):
    #compute the energy of each frame
    energy = np.sum(log_mel_spectrogram, axis=0)
    
    #create binary representation based on the threshold
    binary_representation = np.where(energy > threshold, 1, 0)

    return binary_representation

def apply_median_filter(binary_representation, size=3):
    #apply median filter to the binary representation
    filtered_representation = median_filter(binary_representation, size=size)
    
    return filtered_representation

def find_word_boundaries(filtered_representation):
    word_boundaries = []
    in_word = False
    start_idx = 0
    
    for i, val in enumerate(filtered_representation):
        if val == 1 and not in_word:
            #start of a new word
            start_idx = i
            in_word = True
        elif val == 0 and in_word:
            #end of the current word
            word_boundaries.append((start_idx, i - 1))
            in_word = False
    
    #if the representation ends while still in a word
    if in_word:
        word_boundaries.append((start_idx, len(filtered_representation) - 1))
    
    return word_boundaries

def prepare_data(audio_paths, target_length):
    data = []
    for path in audio_paths:
        log_melspectrogram = extract_melspectrogram(path, hop_length=512, n_fft=2048)
        threshold = np.mean(np.sum(log_melspectrogram, axis=0))
        binary_representation = compute_binary_representation(log_melspectrogram, threshold)
        filtered_representation = apply_median_filter(binary_representation, size=3)

        #ensure consistent length
        if target_length:
            if len(filtered_representation) > target_length:
                filtered_representation = filtered_representation[:target_length]
            elif len(filtered_representation) < target_length:
                filtered_representation = np.pad(filtered_representation, (0, target_length - len(filtered_representation)))

        data.append(filtered_representation)

    return data

def prepare_data_rnn(audio_paths, target_length):
    data = []
    for path in audio_paths:
        log_melspectrogram = extract_melspectrogram(path, hop_length=512, n_fft=2048)
        threshold = np.mean(np.sum(log_melspectrogram, axis=0))
        binary_representation = compute_binary_representation(log_melspectrogram, threshold)
        filtered_representation = apply_median_filter(binary_representation, size=3)
        if len(filtered_representation) > target_length:
            filtered_representation = filtered_representation[:target_length]
        elif len(filtered_representation) < target_length:
            filtered_representation = np.pad(filtered_representation, (0, target_length - len(filtered_representation)))
        data.append(filtered_representation)
    return np.array(data)


'''
Start blah blah
'''

# Paths to 6 audio files
audio_paths = ['./sounds/audio1.wav', './sounds/audio2.wav', './sounds/audio7.m4a',
               './sounds/audio4.mp3', './sounds/audio8.m4a', './sounds/audio6.wav']

# Example labels for the audio files (0 or 1)
labels = [0, 1, 0, 1, 0, 1]

# Prepare data for SVM, Least Squares, and MLP classifiers
data = prepare_data(audio_paths, target_length=2000)
X = np.array(data)
y = np.array(labels)
X = X.reshape(X.shape[0], -1)

# Train SVM classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X, y)
y_pred_svm = svm_classifier.predict(X)
print("SVM Classification Report:")
print(classification_report(y, y_pred_svm))

# Train Least Squares classifier
least_squares_classifier = LinearRegression()
least_squares_classifier.fit(X, y)
y_pred_ls = least_squares_classifier.predict(X)
y_pred_ls_binary = np.where(y_pred_ls > 0.5, 1, 0)
print("Least Squares Classification Report:")
print(classification_report(y, y_pred_ls_binary))

# Train MLP classifier with three layers
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100, 50, 25), activation='relu', solver='adam', max_iter=1000)
mlp_classifier.fit(X, y)
y_pred_mlp = mlp_classifier.predict(X)
print("MLP Classification Report:")
print(classification_report(y, y_pred_mlp))

# Prepare data for RNN classifier
target_length_rnn = 100  # Define target length for RNN
X_rnn = prepare_data_rnn(audio_paths, target_length_rnn).reshape(len(audio_paths), target_length_rnn, 1)
y_rnn = np.array(labels)

# Define RNN model
model_rnn = Sequential([
    SimpleRNN(units=32, input_shape=(target_length_rnn, 1)),
    Dense(2, activation='softmax')
])

# Compile the model
model_rnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model_rnn.fit(X_rnn, y_rnn, epochs=10, batch_size=1, verbose=1)

# Example usage with a new audio file
new_audio_path = './sounds/audio11.m4a'

log_melspectrogram_new = extract_melspectrogram(new_audio_path, hop_length=512, n_fft=2048)
if log_melspectrogram_new is not None:
    threshold_new = np.mean(np.sum(log_melspectrogram_new, axis=0))
    binary_representation_new = compute_binary_representation(log_melspectrogram_new, threshold_new)
    filtered_representation_new = apply_median_filter(binary_representation_new, size=3)

    if len(filtered_representation_new) < 2000:
        filtered_representation_new = np.pad(filtered_representation_new, (0, 2000 - len(filtered_representation_new)))
    elif len(filtered_representation_new) > 2000:
        filtered_representation_new = filtered_representation_new[:2000]

    X_new = filtered_representation_new.reshape(1, -1)

    y_new_pred = svm_classifier.predict(X_new)
    print(f"Predicted label with SVM for the new audio file: {y_new_pred[0]}")

    y_new_pred_ls = least_squares_classifier.predict(X_new)
    y_new_pred_ls_binary = np.where(y_new_pred_ls > 0.5, 1, 0)
    print(f"Predicted label with Least Squares for the new audio file: {y_new_pred_ls_binary[0]}")

    y_new_pred_mlp = mlp_classifier.predict(X_new)
    print(f"Predicted label with MLP for the new audio file: {y_new_pred_mlp[0]}")

    # Ensure the RNN input has the correct shape
    if len(filtered_representation_new) > target_length_rnn:
        filtered_representation_new_rnn = filtered_representation_new[:target_length_rnn]
    elif len(filtered_representation_new) < target_length_rnn:
        filtered_representation_new_rnn = np.pad(filtered_representation_new, (0, target_length_rnn - len(filtered_representation_new)))

    X_new_rnn = filtered_representation_new_rnn.reshape(1, target_length_rnn, 1)
    y_new_pred_rnn = model_rnn.predict(X_new_rnn)
    print(f"Predicted label with RNN for the new audio file: {np.argmax(y_new_pred_rnn[0])}")

# Calculate performance metrics for all classifiers on the training data
classifiers = {
    "SVM": (svm_classifier, y_pred_svm),
    "Least Squares": (least_squares_classifier, y_pred_ls_binary),
    "MLP": (mlp_classifier, y_pred_mlp),
    "RNN": (model_rnn, np.argmax(model_rnn.predict(X_rnn), axis=1))
}

for name, (clf, y_pred) in classifiers.items():
    accuracy = accuracy_score(y, y_pred)
    print(f"{name} Accuracy: {accuracy}")
    print(f"{name} Classification Report:")
    print(classification_report(y, y_pred))