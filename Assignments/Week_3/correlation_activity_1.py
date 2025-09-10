import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Audio file path

audio_file = "CorrelationActivity/Ihavecos200HzPhaseShiftfractionsPi.wav"  # Replace with the actual file path
# Import and load audio file
sample_rate, audio_data = wavfile.read(audio_file)
# Convert to mono if stereo
if len(audio_data.shape) > 1:
    audio_data = np.mean(audio_data, axis=1)
# Normalize amplitude
audio_data = audio_data / max(audio_data)

# Frequency range from 50 Hz to 400 Hz
frequencies = range(50, 401)
func = []
corr = []

# Time vector for the audio signal
t = np.arange(len(audio_data)) / sample_rate

for k, freq in enumerate(frequencies):
    
    cosine_func = np.cos(2 * np.pi * freq * t)
    func.append(cosine_func)
    # Correlate with audio data
    corr.append(np.dot(func[k], audio_data))


plt.figure(figsize=(10, 6))
plt.stem(frequencies, corr)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Correlation")
plt.title("Correlation with Cosines (50-400 Hz)")
plt.grid(True)
plt.show()
