import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
import librosa
import os
import math

from audio_blocking_reference import block_audio


class AudioFeatureExtractor:
    def __init__(self, target_sr=22050, frame_size=2048, hop_ratio=0.5):

        self.target_sr = target_sr
        self.frame_size = frame_size
        self.hop_ratio = hop_ratio
        self.hop_size = int(hop_ratio * frame_size)

    def preprocess_audio(self, audio_path):

        sr, audio = wavfile.read(audio_path)

        # Convert to float
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        elif audio.dtype == np.uint8:
            audio = (audio.astype(np.float32) - 128) / 128.0

        # Convert to mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Resample to target sampel rate
        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)

        # Peak normalization
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))

        return audio

    def compute_stft(self, audio):

        # Get audio blocks from audio_blocking_reference.py
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

    def spectral_centroid(self, magnitude_spectrum, freqs):

        # avoid division by zero for silent frames
        total_energy = np.sum(magnitude_spectrum, axis=1)
        total_energy[total_energy == 0] = 1e-10

        # Weighted average frequency
        centroids = (
            np.sum(magnitude_spectrum * freqs[np.newaxis, :], axis=1) / total_energy
        )
        return centroids

    def spectral_spread(self, magnitude_spectrum, freqs, centroids):

        total_energy = np.sum(magnitude_spectrum, axis=1)
        total_energy[total_energy == 0] = 1e-10

        # spectral spread for each frame
        spreads = []
        for i in range(len(centroids)):
            freq_diff_squared = (freqs - centroids[i]) ** 2
            spread = np.sqrt(
                np.sum(magnitude_spectrum[i] * freq_diff_squared) / total_energy[i]
            )
            spreads.append(spread)

        return np.array(spreads)

    def spectral_flux(self, magnitude_spectrum):

        flux = np.zeros(magnitude_spectrum.shape[0])

        for i in range(1, magnitude_spectrum.shape[0]):
            # difference between consecutive frames
            diff = magnitude_spectrum[i] - magnitude_spectrum[i - 1]
            # sum positive differences (
            flux[i] = np.sum(np.maximum(0, diff))

        return flux

    def mel_filterbank(self, n_filters=13, fmin=0, fmax=None):

        if fmax is None:
            fmax = self.target_sr / 2

        # mel scale conversion
        def hz_to_mel(f):
            return 2595 * np.log10(1 + f / 700)

        def mel_to_hz(m):
            return 700 * (10 ** (m / 2595) - 1)

        # defining mels
        mel_min = hz_to_mel(fmin)
        mel_max = hz_to_mel(fmax)
        mel_points = np.linspace(mel_min, mel_max, n_filters + 2)
        hz_points = mel_to_hz(mel_points)

        # fft bin numbers
        n_fft_bins = self.frame_size // 2 + 1
        freqs = np.linspace(0, self.target_sr / 2, n_fft_bins)
        bin_points = np.floor(
            (self.frame_size + 1) * hz_points / self.target_sr
        ).astype(int)

        # filterbanks
        filterbank = np.zeros((n_filters, n_fft_bins))

        for i in range(1, n_filters + 1):
            left = bin_points[i - 1]
            center = bin_points[i]
            right = bin_points[i + 1]

            # Left slope starts from the left bin and goes to the center bin
            for j in range(left, center):
                if right > left:
                    filterbank[i - 1, j] = (j - left) / (center - left)

            # Right slope starts from the center bin and goes to the right bin
            for j in range(center, right):
                if right > center:
                    filterbank[i - 1, j] = (right - j) / (right - center)

        return filterbank

    def compute_mfcc(self, magnitude_spectrum, n_mfcc=13):

        # Get mel filterbank
        mel_filters = self.mel_filterbank(n_filters=n_mfcc)

        # Apply filterbank to magnitude spectrum
        mel_spectrum = np.dot(magnitude_spectrum, mel_filters.T)

        # Apply log (add small delta to avoid log(0))
        log_mel_spectrum = np.log(mel_spectrum + 1e-10)

        # Apply DCT
        mfcc = np.zeros((log_mel_spectrum.shape[0], n_mfcc))

        for i in range(n_mfcc):
            for j in range(log_mel_spectrum.shape[0]):
                for k in range(n_mfcc):
                    mfcc[j, i] += log_mel_spectrum[j, k] * np.cos(
                        np.pi * i * (k + 0.5) / n_mfcc
                    )

        return mfcc

    def normalize_features(self, features):

        if features.ndim == 1:
            mean_val = np.mean(features)
            std_val = np.std(features)
            if std_val > 0:
                return (features - mean_val) / std_val
            else:
                return features - mean_val
        else:
            normalized = np.zeros_like(features)
            for i in range(features.shape[1]):
                col = features[:, i]
                mean_val = np.mean(col)
                std_val = np.std(col)
                if std_val > 0:
                    normalized[:, i] = (col - mean_val) / std_val
                else:
                    normalized[:, i] = col - mean_val
            return normalized

    def extract_features(self, audio_path):

        print(f"Processing: {audio_path}")

        # Step 0: Preprocess audio
        audio = self.preprocess_audio(audio_path)

        # Step 1: Compute STFT
        magnitude_spectrum, freqs, times = self.compute_stft(audio)

        # Step 2: Compute features
        centroids = self.spectral_centroid(magnitude_spectrum, freqs)
        spreads = self.spectral_spread(magnitude_spectrum, freqs, centroids)
        flux = self.spectral_flux(magnitude_spectrum)
        mfcc = self.compute_mfcc(magnitude_spectrum)

        # Normalize features
        centroids_norm = self.normalize_features(centroids)
        spreads_norm = self.normalize_features(spreads)
        flux_norm = self.normalize_features(flux)
        mfcc_norm = self.normalize_features(mfcc)

        return {
            "spectral_centroid": centroids_norm,
            "spectral_spread": spreads_norm,
            "spectral_flux": flux_norm,
            "mfcc": mfcc_norm,
            "times": times,
            "filename": os.path.basename(audio_path),
        }

    def visualize_features(self, features_list):  ### GENERATED FROM CLAUDE ###

        n_files = len(features_list)

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Audio Feature Analysis", fontsize=16)

        colors = ["blue", "red", "green", "orange"]

        # Plot Spectral Centroid
        ax = axes[0, 0]
        for i, features in enumerate(features_list):
            ax.plot(
                features["times"],
                features["spectral_centroid"],
                color=colors[i % len(colors)],
                label=features["filename"],
                alpha=0.7,
                linewidth=1.5,
            )
        ax.set_title("Spectral Centroid - Musical Brightness")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Normalized Brightness")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

        # Plot Spectral Spread
        ax = axes[0, 1]
        for i, features in enumerate(features_list):
            ax.plot(
                features["times"],
                features["spectral_spread"],
                color=colors[i % len(colors)],
                label=features["filename"],
                alpha=0.7,
                linewidth=1.5,
            )
        ax.set_title("Spectral Spread - Tonal vs Noise-like")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Normalized Spread")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

        # Plot Spectral Flux
        ax = axes[1, 0]
        for i, features in enumerate(features_list):
            ax.plot(
                features["times"],
                features["spectral_flux"],
                color=colors[i % len(colors)],
                label=features["filename"],
                alpha=0.7,
                linewidth=1.5,
            )
        ax.set_title("Spectral Flux - Rhythmic Activity")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Normalized Flux")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

        # Plot MFCC (first coefficient represents overall spectral energy)
        ax = axes[1, 1]
        for i, features in enumerate(features_list):
            ax.plot(
                features["times"],
                features["mfcc"][:, 0],
                color=colors[i % len(colors)],
                label=features["filename"],
                alpha=0.7,
                linewidth=1.5,
            )
        ax.set_title("MFCC[0] - Spectral Energy")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Normalized MFCC[0]")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Additional MFCC visualization
        self.plot_mfcc_heatmap(features_list)

    def plot_mfcc_heatmap(self, features_list):
        """
        Create heatmap visualization for MFCC coefficients

        Shows timbral evolution over time:
        - MFCC[0]: Overall spectral energy
        - MFCC[1-2]: Spectral slope and peaks
        - MFCC[3+]: Finer spectral details
        """
        n_files = len(features_list)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("MFCC Coefficients Heatmap", fontsize=16)

        for i, features in enumerate(features_list):
            if i >= 4:  # Only show first 4 files
                break

            ax = axes[i // 2, i % 2]

            # Transpose for proper orientation (time on x-axis, coefficients on y-axis)
            mfcc_data = features["mfcc"].T

            im = ax.imshow(
                mfcc_data,
                aspect="auto",
                origin="lower",
                cmap="viridis",
                extent=[
                    features["times"][0],
                    features["times"][-1],
                    0,
                    mfcc_data.shape[0],
                ],
            )
            ax.set_title(f"Timbral Evolution - {features['filename']}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("MFCC Coefficient")
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Normalized Amplitude")

        # Hide unused subplots
        for i in range(n_files, 4):
            axes[i // 2, i % 2].set_visible(False)

        plt.tight_layout()
        plt.show()


def main():

    extractor = AudioFeatureExtractor(target_sr=22050, frame_size=2048, hop_ratio=0.5)

    # get all audio files
    folder_path = "./FourInstrumentFiles/"
    audio_files = []

    # read all .wav files from the folder ----- GENERATED FROM CLAUDE -----
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(".wav"):
                audio_files.append(os.path.join(folder_path, filename))

    # Extract features for all files
    all_features = []

    for audio_file in audio_files:
        if os.path.exists(audio_file):
            features = extractor.extract_features(audio_file)
            all_features.append(features)
        else:
            print(f"Warning: File {audio_file} not found!")

    if all_features:
        # Visualize results
        extractor.visualize_features(all_features)

    else:
        print("No audio files found")


if __name__ == "__main__":
    main()
