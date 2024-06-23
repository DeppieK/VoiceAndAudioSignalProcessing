import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
from pydub import AudioSegment
from pydub.playback import play



'''
Melspectorgram
'''
def extract_melspectrogram(audio_path, n_mels=80, hop_length=512, n_fft=2048):

    #load the audio file
    scale, sr = librosa.load(audio_path, sr=None)

    #compute the mel-spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=scale, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)

    #convert to log scale (dB)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

    #print the melspectorgram
    #plt.figure(figsize=(25, 10))
    #librosa.display.specshow(log_mel_spectrogram, 
    #                         x_axis="time",
    #                         y_axis="mel", 
    #                         sr=sr)
    #plt.colorbar(format="%+2.f dB")
    #plt.title("Log Mel Spectrogram")
    #plt.show()

    return log_mel_spectrogram


'''
Binary representation (energy function)
'''
def compute_binary_representation(log_mel_spectrogram, threshold):

    #compute the energy of each frame
    energy = np.sum(log_mel_spectrogram, axis=0)
    
    #create binary representation based on the threshold
    binary_representation = np.where(energy > threshold, 1, 0)

    return binary_representation


'''
Median filter
'''
def apply_median_filter(binary_representation, size=3):

    #apply median filter to the binary representation
    filtered_representation = median_filter(binary_representation, size=size)
    
    return filtered_representation

'''
Word boundaries
'''
def find_word_boundaries(filtered_representation):

    word_boundaries = []
    in_word = False
    start_idx = 0
    
    #loop to find the boundaries
    for i, val in enumerate(filtered_representation):
        if val == 1 and not in_word: #if 1 then foreground sound (word)
            #start of a new word
            start_idx = i
            in_word = True

        elif val == 0 and in_word: #else background sound
            #end of the current word
            word_boundaries.append((start_idx, i - 1))
            in_word = False
    
    #if the representation ends in a word
    if in_word:
        word_boundaries.append((start_idx, len(filtered_representation) - 1))
    
    return word_boundaries


'''
Prepare data for SVM, MLP, RNN and Least Squares
'''
def prepare_data(audio_paths, target_length, return_numpy):

    data = []
    for path in audio_paths:
        #extract melspectogram --> compute binary representation --> apply median filter
        log_melspectrogram = extract_melspectrogram(path, hop_length=512, n_fft=2048)
        threshold = np.mean(np.sum(log_melspectrogram, axis=0))
        binary_representation = compute_binary_representation(log_melspectrogram, threshold)
        filtered_representation = apply_median_filter(binary_representation, size=3)

        if len(filtered_representation) > target_length:
            filtered_representation = filtered_representation[:target_length]
        elif len(filtered_representation) < target_length:
            filtered_representation = np.pad(filtered_representation, (0, target_length - len(filtered_representation)))
        data.append(filtered_representation)

    #if data for RNN then return a numpy array
    if return_numpy:
        return np.array(data)
    
    #else return a list
    return data

'''
αναπραγωγη λεξεων 
'''
# Define a function to play the audio segments corresponding to the word boundaries
def play_detected_words(audio_path, word_boundaries):
    audio = AudioSegment.from_file(audio_path)

    for start, end in word_boundaries:
        # Convert milliseconds to seconds
        start_sec = start * 512 / 22050  # hop_length / sr
        end_sec = end * 512 / 22050  # hop_length / sr

        # Extract the segment of the audio corresponding to the word
        word_segment = audio[int(start_sec * 1000): int(end_sec * 1000)]

        # Play the segment
        play(word_segment)

'''
δευτερο ερωτημα 
'''
def calculate_mean_fundamental_frequency(word_boundaries, log_mel_spectrogram):
    fundamental_frequencies = []

    for start, end in word_boundaries:
        # Extract the frame corresponding to the word boundary
        frame = log_mel_spectrogram[:, start:end+1]

        # Ensure frame is 1-dimensional
        frame = frame.flatten()

        # Calculate short-term autocorrelation
        autocorr = np.correlate(frame, frame, mode='full')

        # Normalize autocorrelation
        autocorr = autocorr / autocorr[len(autocorr) // 2]

        # Set a threshold (e.g., 0.7)
        threshold = 0.7

        # Find indices where autocorrelation is above the threshold
        reliable_indices = np.where(autocorr > threshold)[0]

        # Calculate mean fundamental frequency
        if len(reliable_indices) > 0:
            mean_fundamental_frequency = np.mean(reliable_indices)
        else:
            mean_fundamental_frequency = 0  # or any default value
        
        fundamental_frequencies.append(mean_fundamental_frequency)

    # Calculate average fundamental frequency for all words
    if fundamental_frequencies:
        avg_fundamental_frequency = np.mean(fundamental_frequencies)
    else:
        avg_fundamental_frequency = 0  # or any default value

    return avg_fundamental_frequency


'''
Start
'''

#files for the classifiers' training
audio_paths = ['./sounds/no_speech.wav', './sounds/piano.wav', './sounds/speech.m4a',
               './sounds/drilling.wav', './sounds/no_speech4.wav', './sounds/no_speech3.wav']

#labels of the sounds above --> 0 = no speech/background, 1 = speech/foreground
labels = [0, 1, 1, 1, 0, 0]

#prepare data for SVM, MLP and Least Squares
data = prepare_data(audio_paths, target_length=2000, return_numpy=False)
X = np.array(data)
y = np.array(labels)
X = X.reshape(X.shape[0], -1)

#train SVM
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X, y)
y_pred_svm = svm_classifier.predict(X)
print("SVM Classification Report:")
print(classification_report(y, y_pred_svm))

#train MLP (with three layers)
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100, 50, 25), activation='relu', solver='adam', max_iter=1000)
mlp_classifier.fit(X, y)
y_pred_mlp = mlp_classifier.predict(X)
print("MLP Classification Report:")
print(classification_report(y, y_pred_mlp))

#train Least Squares
least_squares_classifier = LinearRegression()
least_squares_classifier.fit(X, y)
y_pred_ls = least_squares_classifier.predict(X)
y_pred_ls_binary = np.where(y_pred_ls > 0.5, 1, 0) #if y_pred_ls is > 0.5 --> it is classified as 1 (positive class), else it is classified as 0 (negative class)
print("Least Squares Classification Report:")
print(classification_report(y, y_pred_ls_binary))

#prepare data for RNN
target_length_rnn = 100
X_rnn = prepare_data(audio_paths, target_length_rnn, return_numpy=True).reshape(len(audio_paths), target_length_rnn, 1)
y_rnn = np.array(labels)

#define RNN model
model_rnn = Sequential([
    SimpleRNN(units=32, input_shape=(target_length_rnn, 1)),
    Dense(2, activation='softmax')
])

#compile the model
model_rnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#train the rnn model
model_rnn.fit(X_rnn, y_rnn, epochs=10, batch_size=1, verbose=1)

#new audio file
new_audio_path = './sounds/speech2.m4a'

#melspectogram and then binary representation
log_melspectrogram_new = extract_melspectrogram(new_audio_path, hop_length=512, n_fft=2048)
if log_melspectrogram_new is not None:
    threshold_new = np.mean(np.sum(log_melspectrogram_new, axis=0))
    binary_representation_new = compute_binary_representation(log_melspectrogram_new, threshold_new)
    
    #ensure specific length for SVM, MLP and Least Squares
    if len(binary_representation_new) > 2000:
        binary_representation_new = binary_representation_new[:2000]
    elif len(binary_representation_new) < 2000:
        binary_representation_new = np.pad(binary_representation_new, (0, 2000 - len(binary_representation_new)))

    #predict binary representation using trained classifiers
    X_new = binary_representation_new.reshape(1, -1)

    y_new_pred_svm = svm_classifier.predict(X_new)
    y_new_pred_ls = least_squares_classifier.predict(X_new)
    y_new_pred_ls_binary = np.where(y_new_pred_ls > 0.5, 1, 0)
    y_new_pred_mlp = mlp_classifier.predict(X_new)

    #ensure correct shape for RNN
    if len(binary_representation_new) > target_length_rnn:
        binary_representation_new_rnn = binary_representation_new[:target_length_rnn]
    elif len(binary_representation_new) < target_length_rnn:
        binary_representation_new_rnn = np.pad(binary_representation_new, (0, target_length_rnn - len(binary_representation_new)))

    X_new_rnn = binary_representation_new_rnn.reshape(1, target_length_rnn, 1)
    y_new_pred_rnn = model_rnn.predict(X_new_rnn)
    y_new_pred_rnn_binary = np.argmax(y_new_pred_rnn, axis=1).reshape(-1)

    #apply median filter (size = 3)
    filtered_representation_svm = apply_median_filter(y_new_pred_svm, size=3)
    filtered_representation_ls = apply_median_filter(y_new_pred_ls_binary, size=3)
    filtered_representation_mlp = apply_median_filter(y_new_pred_mlp, size=3)
    filtered_representation_rnn = apply_median_filter(y_new_pred_rnn_binary, size=3)

    #print the predicted label for all classifiers
    print(f"Predicted label with SVM for the new audio file: {filtered_representation_svm[0]}")
    print(f"Predicted label with Least Squares for the new audio file: {filtered_representation_ls[0]}")
    print(f"Predicted label with MLP for the new audio file: {filtered_representation_mlp[0]}")
    print(f"Predicted label with RNN for the new audio file: {filtered_representation_rnn[0]}")

    #find word boundaries for the classified representations
    word_boundaries_svm = find_word_boundaries(filtered_representation_svm)
    word_boundaries_ls = find_word_boundaries(filtered_representation_ls)
    word_boundaries_mlp = find_word_boundaries(filtered_representation_mlp)
    word_boundaries_rnn = find_word_boundaries(filtered_representation_rnn)

    #print word boundaries
    print(f"SVM Predicted word boundaries for the new audio file: {word_boundaries_svm}")
    print(f"Least Squares Predicted word boundaries for the new audio file: {word_boundaries_ls}")
    print(f"MLP Predicted word boundaries for the new audio file: {word_boundaries_mlp}")
    print(f"RNN Predicted word boundaries for the new audio file: {word_boundaries_rnn}")

'''
classifiers' comparison
'''
#calculate performance metrics for all classifiers
classifiers = {
    "SVM": (svm_classifier, y_pred_svm),
    "Least Squares": (least_squares_classifier, y_pred_ls_binary),
    "MLP": (mlp_classifier, y_pred_mlp),
    "RNN": (model_rnn, np.argmax(model_rnn.predict(X_rnn), axis=1))
}

#print perfomance metrics
for name, (clf, y_pred) in classifiers.items():
    accuracy = accuracy_score(y, y_pred) #accuracy: y is the true labels and y_pred are the predicted labels.
    print(f"{name} Accuracy: {accuracy}")
    print(f"{name} Classification Report:")
    print(classification_report(y, y_pred))

'''
output the words!!
'''

# Play the detected words
play_detected_words(new_audio_path, word_boundaries_svm)
play_detected_words(new_audio_path, word_boundaries_ls)
play_detected_words(new_audio_path, word_boundaries_mlp)
play_detected_words(new_audio_path, word_boundaries_rnn)


'''
second question
'''
# Assuming you have log_mel_spectrogram and word_boundaries_svm from your existing code
#log_mel_spectrogram = extract_melspectrogram(new_audio_path, hop_length=512, n_fft=2048)
#word_boundaries_svm = find_word_boundaries(filtered_representation_svm)

# Calculate mean fundamental frequency
avg_fundamental_frequency = calculate_mean_fundamental_frequency(word_boundaries_svm, log_melspectrogram_new)

print(f"Average Fundamental Frequency: {avg_fundamental_frequency}")