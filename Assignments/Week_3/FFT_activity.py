import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq

# Audio file path

audio_file = "CorrelationActivity/Ihavesine10s100s&1000sHz.wav"  # Replace with the actual file path
# Import and load audio file
sample_rate, audio_data = wavfile.read(audio_file)
# Convert to mono if stereo
if len(audio_data.shape) > 1:
    audio_data = np.mean(audio_data, axis=1)
# Normalize amplitude
audio_data = audio_data / max(audio_data)

original_length = len(audio_data)

next_power_of_2 = 2 ** int(np.ceil(np.log2(original_length)))  # for the padding

# Pad the signal
padded_audio = np.pad(
    audio_data, (0, next_power_of_2 - original_length), mode="constant"
)
print(f"Padded signal length: {len(padded_audio)}")

fft_result = fft(padded_audio)
frequencies = fftfreq(len(padded_audio), 1 / sample_rate)

magnitude = np.abs(fft_result)
positive_freq_idx = frequencies >= 0
frequencies_positive = frequencies[positive_freq_idx]
magnitude_positive = magnitude[positive_freq_idx]

plt.figure(figsize=(12, 8))

# Plot original signal
plt.subplot(2, 1, 1)
time_original = np.arange(original_length) / sample_rate
plt.plot(time_original, audio_data)
plt.title("Original Audio Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)


plt.subplot(2, 1, 2)
plt.plot(frequencies_positive, magnitude_positive)
plt.title("FFT Magnitude Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 1500)

plt.tight_layout()
plt.show()

# Print peak frequencies
peak_indices = np.argsort(magnitude_positive)[-5:]  # Top 5 peaks
print("\nTop 5 frequency peaks:")
for i, idx in enumerate(reversed(peak_indices)):
    freq = frequencies_positive[idx]
    mag = magnitude_positive[idx]
    print(f"{i+1}. Frequency: {freq:.1f} Hz, Magnitude: {mag:.2f}")
