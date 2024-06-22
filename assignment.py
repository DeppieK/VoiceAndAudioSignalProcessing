import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


def extract_melspectrogram(audio_path, n_mels=80, hop_length=512, n_fft=2048):
    #load the audio file
    scale, sr = librosa.load(audio_path, sr=None)

    #compute the mel-spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=scale, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)

    #convert to log scale (dB)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

    #plt.figure(figsize=(25, 10))
    #librosa.display.specshow(log_mel_spectrogram, 
    #                         x_axis="time",
    #                         y_axis="mel", 
    #                         sr=sr)
    #plt.colorbar(format="%+2.f dB")
    #plt.title("Log Mel Spectrogram")
    #plt.show()

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



'''
start blah blah
'''

#paths to 6 audio files
audio_paths = ['./sounds/audio1.wav', './sounds/audio2.wav', './sounds/audio7.m4a',
               './sounds/audio4.mp3', './sounds/audio8.m4a', './sounds/audio6.wav']

#example labels for the audio files (0 or 1)
labels = [0, 1, 0, 1, 0, 1]  # Example binary labels for the audio files

#prepare data
data = prepare_data(audio_paths, target_length=2000)

#convert data to numpy array
X = np.array(data)
y = np.array(labels)

#flatten the X data
X = X.reshape(X.shape[0], -1)

#train SVM classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X, y)

#make predictions on the training set
y_pred = svm_classifier.predict(X)

#evaluate the classifier
print(classification_report(y, y_pred))


#plot the decision boundaries for visualization (if you have more than 2 features, use PCA or t-SNE for visualization)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='bwr', alpha=0.7)
plt.title("PCA of Training Data")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

# Εκπαίδευση του Least Squares Classifier
least_squares_classifier = LinearRegression()
least_squares_classifier.fit(X, y)

# Πρόβλεψη στο σύνολο εκπαίδευσης
y_pred_ls = least_squares_classifier.predict(X)

# Μετατροπή των συνεχών προβλέψεων σε δυαδικές τιμές (π.χ., > 0.5 -> 1, <= 0.5 -> 0)
y_pred_ls_binary = np.where(y_pred_ls > 0.5, 1, 0)

#classifier evaluation
print(classification_report(y, y_pred_ls_binary))
#example usage with a new audio file
new_audio_path = './sounds/audio11.m4a'

#extract log-mel spectrogram
log_melspectrogram_new = extract_melspectrogram(new_audio_path, hop_length=512, n_fft=2048)

#compute threshold
threshold_new = np.mean(np.sum(log_melspectrogram_new, axis=0))

#compute binary representation
binary_representation_new = compute_binary_representation(log_melspectrogram_new, threshold_new)

#apply median filter
filtered_representation_new = apply_median_filter(binary_representation_new, size=3)

#ensure the dimension matches the trained SVM classifier
if len(filtered_representation_new) < 2000:
    filtered_representation_new = np.pad(filtered_representation_new, (0, 2000 - len(filtered_representation_new)))
elif len(filtered_representation_new) > 2000:
    filtered_representation_new = filtered_representation_new[:2000]

#reshape for prediction
X_new = filtered_representation_new.reshape(1, -1)

#predict using the trained SVM classifier
y_new_pred = svm_classifier.predict(X_new)
print(f"Predicted label for the new audio file: {y_new_pred[0]}")

y_new_pred_ls = least_squares_classifier.predict(X_new)
y_new_pred_ls_binary = np.where(y_new_pred_ls > 0.5, 1, 0)
print(f"Προβλεπόμενη ετικέτα για τον νέο ήχο: {y_new_pred_ls_binary[0]}")