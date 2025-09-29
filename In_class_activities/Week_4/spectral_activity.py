import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load the audio file
audio_file = "sax-phrase.wav"
y, sr = librosa.load(audio_file)

# Normalize the audio
y = librosa.util.normalize(y)

# Calculate the Short-Time Fourier Transform (STFT)
stft = librosa.stft(y)
magnitude = np.abs(stft)

# Calculate time frames for plotting
hop_length = 512  # Default hop length
frames = range(len(magnitude[0]))
times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

# 1. Spectral Centroid
spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

# 2. Spectral Spread (Spectral Bandwidth in librosa)
spectral_spread = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]

# 3. Spectral Flatness
spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]

# 4. Spectral Rolloff
spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]


# Normalize features to 0-1 range for plotting
def normalize_feature(feature):
    return (feature - np.min(feature)) / (np.max(feature) - np.min(feature))


centroid_norm = normalize_feature(spectral_centroid)
spread_norm = normalize_feature(spectral_spread)
flatness_norm = normalize_feature(spectral_flatness)
rolloff_norm = normalize_feature(spectral_rolloff)

# Create the plot
plt.figure(figsize=(12, 10))

# Plot 1: Spectral Centroid
plt.subplot(4, 1, 1)
plt.plot(times, centroid_norm, "b-", linewidth=2)
plt.title("Spectral Centroid")
plt.ylabel("Normalized Value")
plt.grid(True, alpha=0.3)

# Plot 2: Spectral Spread (Bandwidth)
plt.subplot(4, 1, 2)
plt.plot(times, spread_norm, "r-", linewidth=2)
plt.title("Spectral Spread/Bandwidth")
plt.ylabel("Normalized Value")
plt.grid(True, alpha=0.3)

# Plot 3: Spectral Flatness
plt.subplot(4, 1, 3)
plt.plot(times, flatness_norm, "g-", linewidth=2)
plt.title("Spectral Flatness")
plt.ylabel("Normalized Value")
plt.grid(True, alpha=0.3)

# Plot 4: Spectral Rolloff
plt.subplot(4, 1, 4)
plt.plot(times, rolloff_norm, "m-", linewidth=2)
plt.title("Spectral Rolloff")
plt.ylabel("Normalized Value")
plt.xlabel("Time (seconds)")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print some statistics
print("Spectral Features Analysis for sax-phrase.wav")
print("=" * 50)
print(f"Audio duration: {len(y)/sr:.2f} seconds")
print(f"Sample rate: {sr} Hz")
print(f"Number of frames: {len(times)}")
print("\nFeature Statistics (before normalization):")
print(
    f"Spectral Centroid - Mean: {np.mean(spectral_centroid):.2f} Hz, Std: {np.std(spectral_centroid):.2f} Hz"
)
print(
    f"Spectral Spread - Mean: {np.mean(spectral_spread):.2f} Hz, Std: {np.std(spectral_spread):.2f} Hz"
)
print(
    f"Spectral Flatness - Mean: {np.mean(spectral_flatness):.4f}, Std: {np.std(spectral_flatness):.4f}"
)
print(
    f"Spectral Rolloff - Mean: {np.mean(spectral_rolloff):.2f} Hz, Std: {np.std(spectral_rolloff):.2f} Hz"
)

# Optional: If you also have vowels_sung.wav, uncomment the following section
"""
# Load and analyze vowels_sung.wav if it exists
try:
    vowels_file = "vowels_sung.wav"
    y_vowels, sr_vowels = librosa.load(vowels_file)
    y_vowels = librosa.util.normalize(y_vowels)
    
    # Calculate features for vowels_sung.wav
    centroid_vowels = librosa.feature.spectral_centroid(y=y_vowels, sr=sr_vowels)[0]
    spread_vowels = librosa.feature.spectral_bandwidth(y=y_vowels, sr=sr_vowels)[0]
    flatness_vowels = librosa.feature.spectral_flatness(y=y_vowels)[0]
    rolloff_vowels = librosa.feature.spectral_rolloff(y=y_vowels, sr=sr_vowels)[0]
    
    # Create time axis for vowels
    frames_vowels = range(len(centroid_vowels))
    times_vowels = librosa.frames_to_time(frames_vowels, sr=sr_vowels, hop_length=hop_length)
    
    # Normalize features
    centroid_vowels_norm = normalize_feature(centroid_vowels)
    spread_vowels_norm = normalize_feature(spread_vowels)
    flatness_vowels_norm = normalize_feature(flatness_vowels)
    rolloff_vowels_norm = normalize_feature(rolloff_vowels)
    
    # Plot comparison
    plt.figure(figsize=(15, 10))
    
    plt.subplot(4, 1, 1)
    plt.plot(times, centroid_norm, 'b-', linewidth=2, label='sax-phrase.wav')
    plt.plot(times_vowels, centroid_vowels_norm, 'orange', linewidth=2, label='vowels_sung.wav')
    plt.title('Spectral Centroid Comparison (Normalized)')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 1, 2)
    plt.plot(times, spread_norm, 'r-', linewidth=2, label='sax-phrase.wav')
    plt.plot(times_vowels, spread_vowels_norm, 'orange', linewidth=2, label='vowels_sung.wav')
    plt.title('Spectral Spread Comparison (Normalized)')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 1, 3)
    plt.plot(times, flatness_norm, 'g-', linewidth=2, label='sax-phrase.wav')
    plt.plot(times_vowels, flatness_vowels_norm, 'orange', linewidth=2, label='vowels_sung.wav')
    plt.title('Spectral Flatness Comparison (Normalized)')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 1, 4)
    plt.plot(times, rolloff_norm, 'm-', linewidth=2, label='sax-phrase.wav')
    plt.plot(times_vowels, rolloff_vowels_norm, 'orange', linewidth=2, label='vowels_sung.wav')
    plt.title('Spectral Rolloff Comparison (Normalized)')
    plt.ylabel('Normalized Value')
    plt.xlabel('Time (seconds)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nComparison with vowels_sung.wav:")
    print("=" * 50)
    print("Vowels_sung.wav Feature Statistics:")
    print(f"Spectral Centroid - Mean: {np.mean(centroid_vowels):.2f} Hz, Std: {np.std(centroid_vowels):.2f} Hz")
    print(f"Spectral Spread - Mean: {np.mean(spread_vowels):.2f} Hz, Std: {np.std(spread_vowels):.2f} Hz")
    print(f"Spectral Flatness - Mean: {np.mean(flatness_vowels):.4f}, Std: {np.std(flatness_vowels):.4f}")
    print(f"Spectral Rolloff - Mean: {np.mean(rolloff_vowels):.2f} Hz, Std: {np.std(rolloff_vowels):.2f} Hz")
    
except FileNotFoundError:
    print("\nvowels_sung.wav not found - skipping comparison analysis")
"""
