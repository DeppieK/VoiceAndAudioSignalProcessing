import librosa
import numpy as np
import matplotlib.pyplot as plt

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
    
    plt.figure(figsize=(25, 10))
    plt.plot(binary_representation)
    plt.title("Binary Representation (Foreground/Background)")
    plt.xlabel("Frame")
    plt.ylabel("Foreground/Background (1/0)")
    plt.show()

    return binary_representation

'''
start blah blah
'''
audio_path = './sounds/rec7.m4a'
log_melspectrogram = extract_melspectrogram(audio_path, hop_length=512, n_fft=2048)

#define a threshold for the energy
threshold = np.mean(np.sum(log_melspectrogram, axis=0))

binary_representation = compute_binary_representation(log_melspectrogram, threshold)

#print the unique values in the binary representation
print(np.unique(binary_representation))

#print a portion of the binary representation to see details
print(binary_representation[:100]) #to be removed
