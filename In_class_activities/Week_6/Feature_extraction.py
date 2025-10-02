import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa
import os
import pandas as pd
from audio_blocking_reference import block_audio
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
class AudioFeatureExtractor:
    def __init__(
        self, target_sr=22050, frame_size=2048, hop_ratio=0.5, ignore_last_seconds=1.0
    ):
        self.target_sr = target_sr
        self.frame_size = frame_size
        self.hop_ratio = hop_ratio
        self.hop_size = int(hop_ratio * frame_size)
        self.ignore_last_seconds = ignore_last_seconds

########################### PREPROCESSING ###############################

    def preprocess_audio(self, audio_path):
        sr, audio = wavfile.read(audio_path)

        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        elif audio.dtype == np.uint8:
            audio = (audio.astype(np.float32) - 128) / 128.0

        # convert to mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # resample to target sample rate
        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)

        # Remove last few seconds of audio to avoid silence
        samples_to_remove = int(self.ignore_last_seconds * self.target_sr)
        if len(audio) > samples_to_remove:
            audio = audio[:-samples_to_remove]

        # Peak normalization
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))

        return audio

######################## FEATURE COMPUTATION FUNCTIONS ###################################

######################################### STFT #########################################

    def compute_stft(self, audio):
        # Get audio blocks
        blocks, times = block_audio(
            audio,
            sr=self.target_sr,
            frame_size=self.frame_size,
            hop_ratio=self.hop_ratio,
            pad=True,
        )

        # Windowing
        hamming_window = np.hamming(self.frame_size)
        windowed_blocks = blocks * hamming_window[np.newaxis, :]

        # FFT for each block
        stft_result = np.fft.fft(windowed_blocks, axis=1)
        n_fft_bins = self.frame_size // 2 + 1
        stft_result = stft_result[:, :n_fft_bins]

        # Magnitude spectrum
        magnitude_spectrum = np.abs(stft_result)

        # Frequency bins
        freqs = np.fft.fftfreq(self.frame_size, 1 / self.target_sr)[:n_fft_bins]

        return magnitude_spectrum, freqs, times

######################################### SPECTRAL CENTROID #########################################

    def spectral_centroid(self, magnitude_spectrum, freqs):
        # Avoid division by zero for silent frames
        total_energy = np.sum(magnitude_spectrum, axis=1)
        total_energy[total_energy == 0] = 1e-10

        # Weighted average frequency
        centroids = (
            np.sum(magnitude_spectrum * freqs[np.newaxis, :], axis=1) / total_energy
        )
        return centroids

######################################### SPECTRAL SPREAD #########################################

    def spectral_spread(self, magnitude_spectrum, freqs, centroids):
        total_energy = np.sum(magnitude_spectrum, axis=1)
        total_energy[total_energy == 0] = 1e-10

        # Spectral spread for each frame
        spreads = []
        for i in range(len(centroids)):
            freq_diff_squared = (freqs - centroids[i]) ** 2
            spread = np.sqrt(
                np.sum(magnitude_spectrum[i] * freq_diff_squared) / total_energy[i]
            )
            spreads.append(spread)

        return np.array(spreads)

######################################### SPECTRAL FLUX #########################################

    def spectral_flux(self, magnitude_spectrum):

        flux = np.zeros(magnitude_spectrum.shape[0])

        # L2 normalize each frame's magnitude spectrum
        normalized_spectrum = np.zeros_like(magnitude_spectrum)
        for i in range(magnitude_spectrum.shape[0]):
            l2_norm = np.linalg.norm(magnitude_spectrum[i])
            if l2_norm > 0:
                normalized_spectrum[i] = magnitude_spectrum[i] / l2_norm
            else:
                normalized_spectrum[i] = magnitude_spectrum[i]

        # Compute flux from normalized spectra
        for i in range(1, normalized_spectrum.shape[0]):
            diff = normalized_spectrum[i] - normalized_spectrum[i - 1]
            flux[i] = np.sum(np.maximum(0, diff))

        return flux

######################################### SPECTRAL ROLLOFF #########################################

    def spectral_rolloff(self, magnitude_spectrum, freqs, percentile=0.85):

        rolloff = np.zeros(magnitude_spectrum.shape[0])

        for i in range(magnitude_spectrum.shape[0]):
            # Compute cumulative energy - add up the energy of the spectrum up to the current frequency
            cumsum = np.cumsum(magnitude_spectrum[i])

            # frequency where cumsum exceeds percentile of total energy
            total_energy = cumsum[-1]
            if total_energy > 0:
                threshold = percentile * total_energy
                rolloff_bin = np.where(cumsum >= threshold)[0]

                if len(rolloff_bin) > 0:
                    rolloff[i] = freqs[rolloff_bin[0]]
                else:
                    rolloff[i] = freqs[-1]
            else:
                rolloff[i] = 0

        return rolloff

######################################### SPECTRAL FLATNESS #########################################

    def spectral_flatness(self, magnitude_spectrum):

        flatness = np.zeros(magnitude_spectrum.shape[0])

        for i in range(magnitude_spectrum.shape[0]):

            spectrum_flatness = magnitude_spectrum[i] + 1e-10  # avoid division by zero

            geometric_mean = np.exp(np.mean(np.log(spectrum_flatness)))

            arithmetic_mean = np.mean(spectrum_flatness)

            if arithmetic_mean > 0:
                flatness[i] = geometric_mean / arithmetic_mean
            else:
                flatness[i] = 0

        return flatness

######################################### ZERO CROSSING RATE #########################################

    def zero_crossing_rate(self, audio):

        # frames from block_audio
        blocks, times = block_audio(
            audio,
            sr=self.target_sr,
            frame_size=self.frame_size,
            hop_ratio=self.hop_ratio,
            pad=True,
        )

        zcr = np.zeros(blocks.shape[0])

        for i in range(blocks.shape[0]):
            frame = blocks[i]
            # Count zero crossings
            zero_crossings = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
            # Normalize by frame length
            zcr[i] = zero_crossings / self.frame_size

        return zcr

################################ MEL FILTERBANK CALCULATION ################################### (helper for MFCC extraction)

    def mel_filterbank(self, n_filters=13, fmin=0, fmax=None):
        if fmax is None:
            fmax = self.target_sr / 2

        # Mel scale conversion
        def hz_to_mel(f):
            return 2595 * np.log10(1 + f / 700)

        def mel_to_hz(m):
            return 700 * (10 ** (m / 2595) - 1)

        # Defining mels
        mel_min = hz_to_mel(fmin)
        mel_max = hz_to_mel(fmax)
        mel_points = np.linspace(mel_min, mel_max, n_filters + 2)
        hz_points = mel_to_hz(mel_points)

        # FFT bin numbers
        n_fft_bins = self.frame_size // 2 + 1
        freqs = np.linspace(0, self.target_sr / 2, n_fft_bins)
        bin_points = np.floor(
            (self.frame_size + 1) * hz_points / self.target_sr
        ).astype(int)

        # Filterbanks
        filterbank = np.zeros((n_filters, n_fft_bins))

        for i in range(1, n_filters + 1):
            left = bin_points[i - 1]
            center = bin_points[i]
            right = bin_points[i + 1]

            # Left slope
            for j in range(left, min(center, n_fft_bins)):
                if center > left:
                    filterbank[i - 1, j] = (j - left) / (center - left)

            # Right slope
            for j in range(center, min(right, n_fft_bins)):
                if right > center:
                    filterbank[i - 1, j] = (right - j) / (right - center)

        return filterbank

######################################### MFCC EXTRACTION #########################################

    def compute_mfcc(self, magnitude_spectrum, n_mfcc=13):
        # Get mel filterbank
        mel_filters = self.mel_filterbank(n_filters=n_mfcc)

        mel_spectrum = np.dot(magnitude_spectrum, mel_filters.T)

        log_mel_spectrum = np.log(mel_spectrum + 1e-10)

        # DCT
        mfcc = np.zeros((log_mel_spectrum.shape[0], n_mfcc))

        for i in range(n_mfcc):
            for j in range(log_mel_spectrum.shape[0]):
                for k in range(n_mfcc):
                    mfcc[j, i] += log_mel_spectrum[j, k] * np.cos(
                        np.pi * i * (k + 0.5) / n_mfcc
                    )

        return mfcc

######################################### APPLY CMVN TO MFCCS #########################################

    def apply_cmvn(self, mfcc):

        # Calculate mean and std for each coefficient across all frames
        mean = np.mean(mfcc, axis=0, keepdims=True)
        std = np.std(mfcc, axis=0, keepdims=True)

        # Avoid division by zero
        std[std == 0] = 1e-10

        # Normalize
        mfcc_normalized = (mfcc - mean) / std

        return mfcc_normalized

######################################### EXTRACT FEATURES #########################################

    def extract_features(self, audio_path):

        # Step 0: Preprocess audio
        audio = self.preprocess_audio(audio_path)

        # Step 1: Compute STFT
        magnitude_spectrum, freqs, times = self.compute_stft(audio)

        # Step 2: Compute all features
        centroids = self.spectral_centroid(magnitude_spectrum, freqs)
        spreads = self.spectral_spread(magnitude_spectrum, freqs, centroids)
        rolloff = self.spectral_rolloff(magnitude_spectrum, freqs)
        flatness = self.spectral_flatness(magnitude_spectrum)
        flux = self.spectral_flux(magnitude_spectrum)  # L2 normalized
        zcr = self.zero_crossing_rate(audio)  # Computed from time domain
        mfcc = self.compute_mfcc(magnitude_spectrum)

        # Apply CMVN to MFCCs
        mfcc_cmvn = self.apply_cmvn(mfcc)

        # Keep only first 10 MFCC coefficients
        mfcc_cmvn = mfcc_cmvn[:, :10]

        # AGGREGATE: Take mean of each feature (single value per feature)
        # Include standard deviation for each feature for more information
        feature_vector = [
            np.mean(centroids),      # 1. Spectral centroid mean
            np.std(centroids),       # 2. Spectral centroid std
            np.mean(spreads),        # 3. Spectral spread mean
            np.std(spreads),         # 4. Spectral spread std
            np.mean(rolloff),        # 5. Spectral rolloff mean
            np.std(rolloff),         # 6. Spectral rolloff std
            np.mean(flatness),       # 7. Spectral flatness mean
            np.std(flatness),        # 8. Spectral flatness std
            np.mean(zcr),            # 9. Zero crossing rate mean
            np.std(zcr),             # 10. Zero crossing rate std
            np.mean(flux),           # 11. Spectral flux mean
            np.std(flux),            # 12. Spectral flux std
        ]

        # Add MFCC features (mean and std of first 10 coefficients)
        for i in range(10):
            feature_vector.append(np.mean(mfcc[:, i]))  # MFCC mean
            feature_vector.append(np.std(mfcc[:, i]))   # MFCC std

        return np.array(feature_vector)  # Return as numpy array (32 values total)

######################################### NORMALIZE ACROSS FEATURES #########################################

    def normalize_across_features(self, feature_matrix):

        normalized = np.zeros_like(feature_matrix)

        for i in range(feature_matrix.shape[1]):
            col = feature_matrix[:, i]
            mean_val = np.mean(col)
            std_val = np.std(col)
            if std_val > 0:
                normalized[:, i] = (col - mean_val) / std_val
            else:
                normalized[:, i] = col - mean_val

        return normalized


######################################### EXTRACT DATASET LABELS #########################################

    def extract_dataset_features(self, folder_path):  # GENERATED WITH CLAUDE

        # Instrument mapping for micro_medlydb dataset
        instrument_map = {"3": "flute", "4": "piano", "6": "trumpet", "7": "violin"}

        all_features = []
        all_labels = []
        all_filenames = []

        # Get all .wav files
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(".wav"):
                    filepath = os.path.join(root, file)

                    # Extract label from filename
                    # Format: "Medley-solos-DB_validation-3_..." or "Medley-solos-DB_test-6_..."
                    if "Medley-solos-DB" in file:
                        try:
                            # Split filename to extract instrument number
                            # Example: "Medley-solos-DB_validation-3_uuid.wav"
                            parts = file.split("-")

                            # Find index of 'validation' or 'test' part
                            for i, part in enumerate(parts):
                                if "validation" in part or "test" in part:
                                    # Next element after dash is the instrument number
                                    instrument_num = parts[i + 1].split("_")[0]
                                    label = instrument_map.get(
                                        instrument_num, "unknown"
                                    )
                                    break
                            else:
                                label = "unknown"
                        except Exception as e:
                            print(f"Warning: Could not parse label from {file}: {e}")
                            label = "unknown"

                    # Extract features
                    feature_vector = self.extract_features(filepath)

                    all_features.append(feature_vector)
                    all_labels.append(label)
                    all_filenames.append(file)

                    # Optional: print progress
                    print(f"Processed: {file} -> {label}")

        if all_features:
            feature_matrix = np.array(all_features)
            normalized_matrix = self.normalize_across_features(feature_matrix)
            return normalized_matrix, all_labels, all_filenames
        else:
            return None, None, None
        
    
    
######################################### COMPUTE CORRELATION MATRIX ######################################### (also plots the highest and lowest correlations)

    def compute_correlation_matrix(self, normalized_matrix, feature_names):

        # Calculate correlation matrix
        corr_matrix = np.corrcoef(normalized_matrix.T)

        # print the correlation matrix
        print("\n=== CORRELATION MATRIX ===\n")

        print("-" * 130)

        non_mfcc_indices = [
            i for i, name in enumerate(feature_names) if not name.startswith("mfcc_")
        ]
        non_mfcc_names = [feature_names[i] for i in non_mfcc_indices]
        # Extract columns for non-MFCC features, but keep all rows
        submatrix = corr_matrix[:, non_mfcc_indices]
        rows = [name if i < 6 else f"mfcc_{i-6}" for i, name in enumerate(feature_names)]

        df = pd.DataFrame(submatrix, index=rows, columns=non_mfcc_names).round(3)

        print(df.to_string())

        print("-" * 130)

        # Find and print highest and lowest correlations
        n_features = len(feature_names)
        correlations = []

        for i in range(n_features):
            for j in range(i + 1, n_features):
                if j in range(i + 1, n_features):
                    is_i_mfcc = feature_names[i].startswith("mfcc_")
                    is_j_mfcc = feature_names[j].startswith("mfcc_")
                    if is_i_mfcc and is_j_mfcc:
                        continue
                correlations.append(
                    {
                        "feature1": feature_names[i],
                        "feature2": feature_names[j],
                        "idx1": i,  # to store the index of the first and second feature that are being compared
                        "idx2": j,
                        "correlation": corr_matrix[i, j],
                    }
                )

        # Sort by absolute values of correlation in descending order
        correlations_sorted = sorted(
            correlations, key=lambda x: abs(x["correlation"]), reverse=True
        )

        print("\n=== TOP 5 HIGHEST CORRELATIONS ===\n")
        for i in range(min(5, len(correlations_sorted))):
            item = correlations_sorted[i]
            print(
                f"{item['feature1']:<20} vs {item['feature2']:<15} : {item['correlation']:7.4f}"
            )

        print("\n=== TOP 5 LOWEST CORRELATIONS ===\n")
        for i in range(max(0, len(correlations_sorted) - 5), len(correlations_sorted)):
            item = correlations_sorted[i]
            print(
                f"{item['feature1']:<20} vs {item['feature2']:<15} : {item['correlation']:7.4f}"
            )

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        highest = correlations_sorted[
            0
        ]  # store the pair of features with the highest correlation
        axes[0].scatter(
            normalized_matrix[:, highest["idx1"]],
            normalized_matrix[:, highest["idx2"]],
            alpha=0.6,
            s=50,
        )
        axes[0].set_xlabel(highest["feature1"])
        axes[0].set_ylabel(highest["feature2"])
        axes[0].set_title(f"Highest Correlation: {highest['correlation']:.4f}")
        axes[0].grid(True, alpha=0.3)

        lowest = correlations_sorted[
            -1
        ]  # store the pair of features with the lowest correlation
        axes[1].scatter(
            normalized_matrix[:, lowest["idx1"]],
            normalized_matrix[:, lowest["idx2"]],
            alpha=0.6,
            s=50,
            color="orange",
        )
        axes[1].set_xlabel(lowest["feature1"])
        axes[1].set_ylabel(lowest["feature2"])
        axes[1].set_title(f"Lowest Correlation: {lowest['correlation']:.4f}")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("correlation_extremes.png", dpi=300, bbox_inches="tight")
        plt.show()

        return corr_matrix


######################################### END OF AUDIO FEATURE EXTRACTION FUNCTIONS ##########################

# =============================================================================================================

def main():

    # Initialize audio feature extractor
    extractor = AudioFeatureExtractor(
        target_sr=22050, frame_size=2048, hop_ratio=0.5, ignore_last_seconds=1.0
    )

    # Paths to training and test datasets
    training_folder_path = "./micro_medlydb/validate/"
    test_folder_path = "./micro_medlydb/test/"

    # Extract training features and labels
    training_features, training_labels, training_filenames = extractor.extract_dataset_features(training_folder_path)

    # Extract test features and labels
    test_features, test_labels, test_filenames = extractor.extract_dataset_features(test_folder_path)

    # Print training feature matrix
    print("\n=== Training Feature Matrix ===\n")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    # rows = files, columns = features
    print(pd.DataFrame(training_features).round(5).to_string(index=False, header=False))

    # Feature names
    feature_names = [
        "spectral_centroid_mean", "spectral_centroid_std",
        "spectral_spread_mean", "spectral_spread_std", 
        "spectral_rolloff_mean", "spectral_rolloff_std",
        "spectral_flatness_mean", "spectral_flatness_std",
        "zero_crossing_rate_mean", "zero_crossing_rate_std",
        "spectral_flux_mean", "spectral_flux_std"
    ]
    for i in range(10):
        feature_names.extend([f"mfcc_{i}_mean", f"mfcc_{i}_std"])

    # Prints correlation matrix and finds least and most correlated features. 
    # ** Close plot to continue script. **
    extractor.compute_correlation_matrix(training_features, feature_names)

    # KNN Classifier
    print("\n=== KNN Classifier ===\n")

    # Use GridSearchCV to find the best k value for the KNN classifier
    k_range = list(range(1, 21))
    param_grid = dict(n_neighbors=k_range)
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
    grid.fit(training_features, training_labels)
    best_k = grid.best_params_['n_neighbors']
    print(f"Optimal K: {best_k}")
    print(f"Cross-Validation Accuracy with Optimal K: {grid.best_score_:.4f}")

    # Use the best k value to train the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(training_features, training_labels)
    predictions = knn.predict(test_features)
    print(f"Predictions: {predictions}")
    print(f"Test Labels: {test_labels}")
    print(f"Accuracy: {accuracy_score(test_labels, predictions)}")
    print(f"Confusion Matrix: \n {confusion_matrix(test_labels, predictions)}")
    print(f"Classification Report: \n {classification_report(test_labels, predictions)}")

if __name__ == "__main__":
    main()
