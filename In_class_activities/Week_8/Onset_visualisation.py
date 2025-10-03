import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal

# Read the audio file
sample_rate, audio_data = wavfile.read('./Onset_Detection/vulpeck_sky_mall.wav')
    
# If stereo, convert to mono
if len(audio_data.shape) > 1:
   audio_data = np.mean(audio_data, axis=1)

audio_data = audio_data/max(audio_data)
audio_data = audio_data.astype(np.float32)

# Create a splice of the audio after listening to it for one onset
audio_splice = audio_data[10:50000]

# Compute spectrogram, experiment with parameters
frequencies, times, spectrogram = signal.spectrogram(audio_splice,fs=sample_rate,window='hann',nperseg=1024,noverlap=512)

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
# Plot waveform
ax1.plot(audio_splice, linewidth=0.5)
ax1.set_xlabel('Samples')
ax1.set_ylabel('Amplitude')
ax1.grid(True, alpha=0.3)
    
# Plot spectrogram
im = ax2.pcolormesh(times, frequencies, 10 * np.log10(spectrogram + 1e-10),
                        shading='gouraud', cmap='viridis')
ax2.set_ylabel('Frequency (Hz)')
ax2.set_xlabel('Time (s)')
ax2.set_title('Spectrogram')
ax2.set_ylim(0, sample_rate / 2)
    
# Add colorbar
cbar = plt.colorbar(im, ax=ax2)
cbar.set_label('Power (dB)')
    
plt.tight_layout()
plt.show()