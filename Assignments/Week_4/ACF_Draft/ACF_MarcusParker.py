# Part 1
# def estimate_f0(audio_frame, sr, minfreq, maxfreq, threshold)
#
#   Pre-processing steps:
#   1) DC offset removal: Subtract the mean of the audio frame from the audio frame.
#   mean_val_frame = np.mean(audio_frame)
#   audio_frame = audio_frame - mean_val_frame
#
#   // 1. Calculate the ACF of the audio_frame.
#   acf = calculate_autocorrelation(audio_frame)
#
#   // 2. Normalize the ACF by dividing by the value at the first value
#   normalized_acf = acf / acf[0]
#
#   // 3. Convert the frequency search range (minfreq, maxfreq) to a lag range
#   min_lag = floor(sr / maxfreq)
#   max_lag = ceil(sr / minfreq)
#
#   // 4. Search for the highest peak in the normalized ACF within the lag range.
#   // Start the search from min_lag to exclude the peak at lag 0.
#   peak_index = find_peak_index_in_range(normalized_acf, min_lag, max_lag)
#   peak_value = normalized_acf[peak_index]
#
#   // 5. Check if the peak value is above the reliability threshold.
#   if peak_value >= threshold
#
#     // 6. If it is, convert the peak's lag (index) back to a frequency. This is our fundamental frequency estimate.
#     estimated_frequency = sr / peak_index
#
#     // 7. Return the estimated frequency.
#     RETURN estimated_frequency
#
#   else
#
#     // 8. If the peak is not reliable, return "not a number" (np.nan).
#     return np.nan
#
from scipy.signal import correlate
from audio_blocking_reference import * # Had to change the name slightly
import matplotlib.pyplot as plt

def estimate_f0(audio_frame, sr, minfreq, maxfreq, threshold):
    # 1. Calculate the ACF of the audio_frame.
    acf = correlate(audio_frame, audio_frame, mode='full')
    acf = acf[len(audio_frame) - 1:]

    # 2. Normalize the ACF by dividing by the value at the first value
    normalized_acf = acf / acf[0]

    # 3. Convert the frequency search range (minfreq, maxfreq) to a lag range
    min_lag = int(np.floor(sr / maxfreq))
    max_lag = int(np.ceil(sr / minfreq))

    # Ensure min_lag and max_lag are within the bounds of normalized_acf
    min_lag = max(1, min_lag)
    max_lag = min(len(normalized_acf) - 1, max_lag)

    # 4. Search for the highest peak in the normalized ACF within the lag range.
    peak = normalized_acf[min_lag : max_lag + 1]

    # Check if peak is empty
    if len(peak) == 0:
        return np.nan

    peak_index_relative = np.argmax(peak)
    peak_index = min_lag + peak_index_relative
    peak_value = normalized_acf[peak_index]

    # 5. Check if the peak value is above the reliability threshold.
    if peak_value >= threshold:
        # 6. If it is, convert the peak's lag (index) back to a frequency. This is our fundamental frequency estimate.
        estimated_frequency = sr / peak_index

        # 7. Return the estimated frequency.
        return estimated_frequency

    else:
        # 8. If the peak is not reliable, return np.nan.
        return np.nan

# Part 2
# Pull data
data_1, times_1 = block_audio("violin-sanidha.wav")
data_2, times_2 = block_audio("pansoori_female.wav")
# Pull sample rates
sr_1, audio_1 = wavfile.read("violin-sanidha.wav")
sr_2, audio_2 = wavfile.read("pansoori_female.wav")
print(sr_1, sr_2)

# Choose frequencies
# 196 to 2637 violin
# 120 to 1000 female pansori singer

# Violin Plotting
estimated_frequencies = [estimate_f0(frame, sr_1, 196, 2637, 0.7) for frame in data_1]

plt.figure(figsize=(15, 7))
plt.scatter(times_1, estimated_frequencies, s=10)
plt.title('Estimated Fundamental Frequency of Violin')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')

plt.ylim(100, 800)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Pansori Singer Plotting
estimated_frequencies = [estimate_f0(frame, sr_2, 120, 1000, 0.85) for frame in data_2]

plt.figure(figsize=(15, 7))
plt.scatter(times_2, estimated_frequencies, s=10)
plt.title('Estimated Fundamental Frequency of Pansori Singer')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')

plt.ylim(100, 800)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()