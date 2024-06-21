import librosa
import numpy as np
import matplotlib.pyplot as plt


def compute_melspectrogram(audio_path, n_mels=80, hop_length=512, n_fft=2048):
    """
    Computes the mel-spectrogram for an audio file.

    Parameters:
    audio_path (str): Path to the audio file.
    n_mels (int): Number of mel bands to generate.
    hop_length (int): Number of samples between successive frames.
    n_fft (int): Length of the FFT window.

    Returns:
    np.ndarray: Mel-spectrogram.
    """

    # Load the audio file
    scale, sr = librosa.load(audio_path, sr=None)

    # Compute the mel-spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=scale, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)

    # Convert to log scale (dB)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

    plt.figure(figsize=(25, 10))
    librosa.display.specshow(mel_spectrogram, 
                        x_axis="time",
                        y_axis="mel", 
                        sr=sr)
    plt.colorbar(format="%+2.f")
    plt.show()

    return log_mel_spectrogram

# Example usage
audio_path = './sounds/piano.wav'
melspectrogram = compute_melspectrogram(audio_path, hop_length=512, n_fft=2048)

print(melspectrogram.shape)

