import librosa
import numpy as np

print(np.__version__)

def compute_melspectrogram(audio_path, hop_length=512, n_fft=2048):
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
    y, sr = librosa.load(audio_path, sr=None)

    # Compute the mel-spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft)

    # Convert to log scale (dB)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    return mel_spectrogram_db

# Example usage
audio_path = '../sounds/739652__phonoflora__drilling.wav'
melspectrogram = compute_melspectrogram(audio_path, hop_length=512, n_fft=2048)

print(melspectrogram.shape)