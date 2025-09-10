########## PSUEDO CODE FOR PITCH ESTIMATION FUNCTION ########## 

# def estimate_f0(audio_frame, sr, minfreq, maxfreq, threshold)

#   /////PRE-PROCESSING STEPS//////

#   1) DC offset removal: Subtract the mean of the audio frame from the audio frame.

#   mean_val_frame = np.mean(audio_frame)
#   audio_frame = audio_frame - mean_val_frame

#   2) Windowing: Apply a window function to the audio frame.
#   window = np.hamming(len(audio_frame))
#   audio_frame = audio_frame * window

#   /////ACF FUNCTION/////

#   Mathematically, The autocorrelation function of a signal x with length N at lag k - 
#   Acf(k) = sum(x[n] * x[n + k]) for n in range(0, N - 1 - k)
#   >>>>The ACF should be calculated at lag 0 to N - 1 for N number of samples<<<<<

#   acf = initialise_array(len(audio_frame))
#   for lag from 0 to len(audio_frame) - 1:
#       main_segment = audio_frame[0:len(audio_frame) - lag]
#       lagged_segment = audio_frame[lag:len(audio_frame)]
#       acf[lag] = dot_product(main_segment, lagged_segment)

#   ////PITCH ESTIMATION FUNCTION////

#   1. Normalising the ACF by dividing by the value at the first value
#   normalised_acf = acf / acf[0]

#   2. Find the search range by converting the frequency search range (minfreq, maxfreq) to a lag range
#   min_lag = int(sr / maxfreq)
#   max_lag = int(sr / minfreq)

#   3.Find the range of search 
#   search_range = normalized_acf[min_lag : max_lag + 1]

#   4. Find the peak index in the search range
#   rel_peak_index = np.argmax(search_range)
#   peak_index = min_lag + relative_peak_index

#   5. Check if the peak value is above the reliability threshold.
#   if peak_value < threshold:
#       return np.nan

#   6. Converting peak index to frequency
#   f0_est = sr / peak_index

#    return estimated_frequency
#############################################################

from scipy.signal import correlate
from audio_blocking_reference import * # Had to change the name slightly
import matplotlib.pyplot as plt
from scipy.io import wavfile

def estimate_f0(audio_frame, sr, minfreq, maxfreq, threshold):
    # ======PREPROCESSING======
    
    # DC offset removal
    mean_val_frame = np.mean(audio_frame)
    audio_frame = audio_frame - mean_val_frame

    # Windowing
    window = np.hamming(len(audio_frame))
    audio_frame = audio_frame * window

    # ======ACF FUNCTION======
    
    N = len(audio_frame)
    acf = np.zeros(N)
    
    for lag in range(N):
        main_segment = audio_frame[0:N - lag]
        lagged_segment = audio_frame[lag:N]
        acf[lag] = np.dot(main_segment, lagged_segment)

    normalized_acf = acf / acf[0]

    # ======PITCH ESTIMATION FUNCTION======
    
    # frequency search range (minfreq, maxfreq) conversion to a lag range
    min_lag = int(sr / maxfreq)
    max_lag = int(sr / minfreq)

    # defining bounds for the search range
    min_lag = max(min_lag, 1)
    max_lag = min(max_lag, N - 1)
    if min_lag >= max_lag:
        return np.nan

    # Defininf search range
    search_range = normalized_acf[min_lag : max_lag + 1]

    # Peak finding
    peak_index_norm = np.argmax(search_range)
    peak_index = min_lag + peak_index_norm
    peak_value = normalized_acf[peak_index]

    # Thresholding
    if peak_value >= threshold:
        estimated_frequency = sr / peak_index
    else:
        return np.nan
    
    return estimated_frequency



# Part 2
# Pull data
data_1, times_1 = block_audio("violin-sanidha.wav")
data_2, times_2 = block_audio("pansoori_female.wav")
data_3, times_3 = block_audio("Sample.wav")
# Pull sample rates
sr_1, audio_1 = wavfile.read("violin-sanidha.wav")
sr_2, audio_2 = wavfile.read("pansoori_female.wav")
sr_3, audio_3 = wavfile.read("Sample.wav")
print(sr_1, sr_2, sr_3)

# Choose frequencies
# 150 to 2700 violin
# 120 to 1500 female pansori singer
# 120 to 2000 custom sample

# Violin Plotting
estimated_frequencies = [estimate_f0(frame, sr_1, 150, 2700, 0.7) for frame in data_1]

plt.figure(figsize=(15, 7))
plt.scatter(times_1, estimated_frequencies, s=10)
plt.title('Estimated Fundamental Frequency of Violin')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')

plt.ylim(100, 800)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Pansori Singer Plotting
estimated_frequencies = [estimate_f0(frame, sr_2, 120, 2000, 0.7) for frame in data_2]

plt.figure(figsize=(15, 7))
plt.scatter(times_2, estimated_frequencies, s=10)
plt.title('Estimated Fundamental Frequency of Pansori Singer')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')

plt.ylim(100, 800)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Plotting for our custom sample
estimated_frequencies = [estimate_f0(frame, sr_3, 120, 2000, 0.7) for frame in data_3]

plt.figure(figsize=(15, 7))
plt.scatter(times_3, estimated_frequencies, s=10)
plt.title('Estimated Fundamental Frequency of Sample')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')

plt.ylim(100, 800)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()