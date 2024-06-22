import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter


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
    
    plt.figure(figsize=(25, 10))
    plt.plot(filtered_representation)
    plt.title("Filtered Binary Representation (Foreground/Background)")
    plt.xlabel("Frame")
    plt.ylabel("Foreground/Background (1/0)")
    plt.show()
    
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
    
    #φf the representation ends while still in a word
    if in_word:
        word_boundaries.append((start_idx, len(filtered_representation) - 1))
    
    return word_boundaries

'''
start blah blah
'''
audio_path = './sounds/piano.wav'
log_melspectrogram = extract_melspectrogram(audio_path, hop_length=512, n_fft=2048)

#define a threshold for the energy
threshold = np.mean(np.sum(log_melspectrogram, axis=0))

binary_representation = compute_binary_representation(log_melspectrogram, threshold)

#apply median filter to remove small errors
filtered_representation = apply_median_filter(binary_representation, size=5)

#print the unique values in the filtered binary representation
print(np.unique(filtered_representation))

#print a portion of the filtered binary representation to see details
print(filtered_representation[:100]) #to be removed

#find word boundaries
word_boundaries = find_word_boundaries(filtered_representation)
print(word_boundaries)

#print or plot the word boundaries on the binary representation
plt.figure(figsize=(25, 10))
plt.plot(filtered_representation)
for start, end in word_boundaries:
    plt.axvspan(start, end, color='red', alpha=0.3)
plt.title("Word Boundaries in Binary Representation")
plt.xlabel("Frame")
plt.ylabel("Foreground/Background (1/0)")
plt.show()