import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

audio_file = "vowels_sung.wav"
y, sr = librosa.load(audio_file, sr=None)

print(f"Audio file loaded: {audio_file}")
print(f"Sample rate: {sr} Hz")
print(f"Duration: {len(y)/sr:.2f} seconds")

# Compute MFCCs - default n_mfcc=13, n_mels=128
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_mels=128, fmax=sr / 2)

# Compute mel spectrogram
mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr / 2)
mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

print(f"MFCC shape: {mfccs.shape}")
print(f"Mel spectrogram shape: {mel_spectrogram_db.shape}")

# Create subplots for both visualizations
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot MFCCs
img1 = librosa.display.specshow(
    mfccs, sr=sr, x_axis="time", y_axis="off", cmap="RdYlBu", ax=ax1
)
ax1.set_title("MFCCs (Mel-Frequency Cepstral Coefficients)")
ax1.set_ylabel("MFCC Coefficient")
plt.colorbar(img1, ax=ax1, label="MFCC Value")

# Plot Mel Spectrogram
img2 = librosa.display.specshow(
    mel_spectrogram_db,
    sr=sr,
    x_axis="time",
    y_axis="mel",
    fmax=sr / 2,
    cmap="viridis",
    ax=ax2,
)
ax2.set_title("Mel Spectrogram")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Mel Frequency")
plt.colorbar(img2, ax=ax2, format="%+2.0f dB", label="Power (dB)")

plt.tight_layout()
plt.show()
