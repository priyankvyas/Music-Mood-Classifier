import librosa
import numpy as np

# Extract Mel-Frequency Cepstral Coefficients for a given .wav file.
# MFCCs capture the timbral features of the audio.
def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, duration=30) # Duration set for consistency
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13) # Typical value for MFCC
    return np.mean(mfcc, axis=1) # Mean of MFCCs for each file

# Extract Chroma for a given .wav file. Chroma captures the harmonic and
# melodic characteristics.
def extract_chroma(file_path):
    y, sr = librosa.load(file_path, duration=30) # Duration set for consistency
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    return np.mean(chroma, axis=1) # Mean of Chromas for each file

# Extract Spectral Contrast for a given .wav file. Spectral Contrast captures
# the difference between peaks and valleys in the spectrum.
def extract_spectral_contrast(file_path):
    y, sr = librosa.load(file_path, duration=30) # Duration set for consistency
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    return np.mean(spectral_contrast, axis=1) # Mean of Spectral Contrasts for each file

# Extract Tonnetz for a given .wav file. Tonnetz or tonal centroid represent
# the tonal distribution in a song.
def extract_tonnetz(file_path):
    y, sr = librosa.load(file_path, duration=30) # Duration set for consistency
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    return np.mean(tonnetz, axis=1) # Mean of Tonnetz for each file

# Extract ZCR for a given .wav file. Zero Crossing Rate or ZCR measures the rate
# rate at which a signal changes sign, useful for distinguishing percussive vs.
# harmonic sounds.
def extract_zcr(file_path):
    y, = librosa.load(file_path, duration=30) # Duration set for consistency
    zcr = librosa.feature.zero_crossing_rate(y)
    return zcr # Zero Crossing Rates for each file

# Extract Spectral Roll-Off for a given .wav file. Spectral Roll-Off indicates 
# the frequency below which a percentage of the total spectral energy is contained.
def extract_rolloff(file_path):
    y, sr = librosa.load(file_path, duration=30) # Duration set for consistency
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    return spectral_rolloff # Spectral Rolloff for each file