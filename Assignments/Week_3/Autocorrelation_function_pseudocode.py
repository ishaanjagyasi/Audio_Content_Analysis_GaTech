"""
Autocorrelation-based Fundamental Pitch Detection
================================================

This module implements autocorrelation-based pitch detection for audio signals
using audio blocking to extract the trajectory of fundamental pitches.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import librosa
import os


def load_audio_file(file_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and return the signal and sample rate.

    Args:
        file_path: Path to the audio file
        target_sr: Target sample rate (will resample if different)

    Returns:
        Tuple of (audio signal, sample rate)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # Load audio file using librosa
    signal, sr = librosa.load(file_path, sr=target_sr)

    print(f"Loaded audio file: {file_path}")
    print(f"Duration: {len(signal) / sr:.2f} seconds")
    print(f"Sample rate: {sr} Hz")
    print(f"Signal shape: {signal.shape}")

    return signal, sr


def audio_blocking(
    signal: np.ndarray, block_size: int, hop_size: int
) -> List[np.ndarray]:
    """
    Split audio signal into overlapping blocks for frame-by-frame analysis.

    Args:
        signal: Input audio signal
        block_size: Size of each block in samples
        hop_size: Number of samples to advance between blocks

    Returns:
        List of audio blocks
    """
    blocks = []
    start = 0

    while start + block_size <= len(signal):
        block = signal[start : start + block_size]
        blocks.append(block)
        start += hop_size

    return blocks


def autocorrelation(signal: np.ndarray, max_lag: Optional[int] = None) -> np.ndarray:
    """
    Compute the autocorrelation function of a signal.

    Args:
        signal: Input signal block
        max_lag: Maximum lag to compute (default: length of signal - 1)

    Returns:
        Autocorrelation values for each lag
    """
    n = len(signal)
    if max_lag is None:
        max_lag = n - 1

    # Normalize the signal (zero mean)
    signal = signal - np.mean(signal)

    # Compute autocorrelation using numpy correlate
    autocorr = np.correlate(signal, signal, mode="full")

    # Take only positive lags and normalize
    autocorr = autocorr[n - 1 : n - 1 + max_lag + 1]
    autocorr = autocorr / autocorr[0]  # Normalize by zero-lag value

    return autocorr


def find_fundamental_pitch(
    signal: np.ndarray,
    sample_rate: int,
    min_freq: float = 80.0,
    max_freq: float = 800.0,
) -> Tuple[float, float]:
    """
    Find the fundamental pitch of a signal block using autocorrelation.

    Args:
        signal: Input signal block
        sample_rate: Sampling rate in Hz
        min_freq: Minimum expected frequency in Hz
        max_freq: Maximum expected frequency in Hz

    Returns:
        Tuple of (frequency in Hz, confidence score)
    """
    # Convert frequency bounds to lag bounds
    min_lag = int(sample_rate / max_freq)
    max_lag = int(sample_rate / min_freq)

    # Ensure we don't exceed signal length
    max_lag = min(max_lag, len(signal) - 1)

    if min_lag >= max_lag:
        return 0.0, 0.0

    # Compute autocorrelation
    autocorr = autocorrelation(signal, max_lag)

    # Find the peak in the expected lag range
    search_range = autocorr[min_lag : max_lag + 1]

    if len(search_range) == 0:
        return 0.0, 0.0

    # Find the lag with maximum autocorrelation
    peak_idx = np.argmax(search_range)
    peak_lag = min_lag + peak_idx
    confidence = search_range[peak_idx]

    # Convert lag to frequency
    if peak_lag > 0:
        frequency = sample_rate / peak_lag
    else:
        frequency = 0.0

    return frequency, confidence


def extract_pitch_trajectory(
    signal: np.ndarray,
    sample_rate: int,
    block_size: int = 2048,
    hop_size: int = 512,
    min_freq: float = 80.0,
    max_freq: float = 800.0,
    confidence_threshold: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract the trajectory of fundamental pitches from an audio signal.

    Args:
        signal: Input audio signal
        sample_rate: Sampling rate in Hz
        block_size: Size of each analysis block
        hop_size: Hop size between blocks
        min_freq: Minimum expected frequency in Hz
        max_freq: Maximum expected frequency in Hz
        confidence_threshold: Minimum confidence for pitch detection

    Returns:
        Tuple of (time_frames, frequencies, confidences)
    """
    # Split signal into blocks
    blocks = audio_blocking(signal, block_size, hop_size)

    frequencies = []
    confidences = []
    time_frames = []

    for i, block in enumerate(blocks):
        # Apply window function to reduce spectral leakage
        windowed_block = block * np.hanning(len(block))

        # Extract pitch
        freq, conf = find_fundamental_pitch(
            windowed_block, sample_rate, min_freq, max_freq
        )

        # Apply confidence threshold
        if conf < confidence_threshold:
            freq = 0.0  # Mark as unvoiced

        frequencies.append(freq)
        confidences.append(conf)

        # Calculate time for this frame
        time_sec = (i * hop_size) / sample_rate
        time_frames.append(time_sec)

    return np.array(time_frames), np.array(frequencies), np.array(confidences)


def plot_pitch_trajectory(
    time_frames: np.ndarray,
    frequencies: np.ndarray,
    confidences: np.ndarray,
    title: str = "Pitch Trajectory",
):
    """
    Plot the extracted pitch trajectory.

    Args:
        time_frames: Time points for each frame
        frequencies: Fundamental frequencies
        confidences: Confidence scores
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot pitch trajectory
    voiced_mask = frequencies > 0
    ax1.plot(
        time_frames[voiced_mask],
        frequencies[voiced_mask],
        "b.-",
        markersize=4,
        linewidth=1,
    )
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Plot confidence
    ax2.plot(time_frames, confidences, "r.-", markersize=3, linewidth=1)
    ax2.axhline(y=0.3, color="gray", linestyle="--", alpha=0.7, label="Threshold")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Confidence")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()


def analyze_audio_pitch(
    audio_file_path: str,
    target_sr: int = 16000,
    block_size: int = 2048,
    hop_size: int = 512,
    min_freq: float = 80.0,
    max_freq: float = 800.0,
    confidence_threshold: float = 0.2,
    show_plots: bool = True,
) -> dict:
    """
    Convenient function to analyze pitch trajectory from any audio file.

    Args:
        audio_file_path: Path to the audio file
        target_sr: Target sampling rate
        block_size: Analysis window size
        hop_size: Step size between windows
        min_freq: Minimum frequency to detect
        max_freq: Maximum frequency to detect
        confidence_threshold: Minimum confidence for pitch detection
        show_plots: Whether to display visualization plots

    Returns:
        Dictionary containing analysis results
    """
    try:
        # Load audio file
        signal, sample_rate = load_audio_file(audio_file_path, target_sr)

        # Extract pitch trajectory
        time_frames, frequencies, confidences = extract_pitch_trajectory(
            signal,
            sample_rate,
            block_size,
            hop_size,
            min_freq,
            max_freq,
            confidence_threshold,
        )

        # Calculate statistics
        voiced_mask = frequencies > 0
        voiced_frames = np.sum(voiced_mask)
        total_frames = len(frequencies)
        voicing_percentage = 100 * voiced_frames / total_frames

        results = {
            "time_frames": time_frames,
            "frequencies": frequencies,
            "confidences": confidences,
            "voiced_frames": voiced_frames,
            "total_frames": total_frames,
            "voicing_percentage": voicing_percentage,
            "duration": len(signal) / sample_rate,
            "sample_rate": sample_rate,
        }

        if voiced_frames > 0:
            voiced_freqs = frequencies[voiced_mask]
            results.update(
                {
                    "avg_frequency": np.mean(voiced_freqs),
                    "min_frequency": np.min(voiced_freqs),
                    "max_frequency": np.max(voiced_freqs),
                    "std_frequency": np.std(voiced_freqs),
                }
            )

        # Display results
        print(f"\nPitch Analysis Results for: {audio_file_path}")
        print(f"Duration: {results['duration']:.2f} seconds")
        print(f"Total frames: {total_frames}")
        print(f"Voiced frames: {voiced_frames} ({voicing_percentage:.1f}%)")

        if voiced_frames > 0:
            print(f"Average frequency: {results['avg_frequency']:.1f} Hz")
            print(
                f"Frequency range: {results['min_frequency']:.1f} - {results['max_frequency']:.1f} Hz"
            )
            print(f"Standard deviation: {results['std_frequency']:.1f} Hz")

        # Show plots if requested
        if show_plots:
            plot_pitch_trajectory(
                time_frames,
                frequencies,
                confidences,
                f"Pitch Analysis: {audio_file_path}",
            )

        return results

    except Exception as e:
        print(f"Error analyzing audio file: {e}")
        return None


# Example usage and test
if __name__ == "__main__":
    print("=" * 60)
    print("AUTOCORRELATION-BASED PITCH DETECTION SYSTEM")
    print("=" * 60)

    # Method 1: Using the convenient function
    audio_file = "Analog_test.wav"
    print(f"\n📊 Analyzing audio file: {audio_file}")
    print("-" * 50)

    results = analyze_audio_pitch(
        audio_file_path=audio_file,
        target_sr=16000,
        block_size=2048,
        hop_size=512,
        min_freq=80,
        max_freq=800,
        confidence_threshold=0.2,
        show_plots=True,
    )

    if results:
        print("\n✅ Analysis completed successfully!")
        print(
            f"🎵 Found {results['voiced_frames']} voiced frames out of {results['total_frames']} total frames"
        )

    # Method 2: Manual analysis for comparison (optional)
    print("\n" + "=" * 60)
    print("MANUAL ANALYSIS EXAMPLE")
    print("=" * 60)

    try:
        # Load and analyze manually for demonstration
        signal, sample_rate = load_audio_file(audio_file, target_sr=16000)

        # Extract pitch trajectory with different parameters
        time_frames, frequencies, confidences = extract_pitch_trajectory(
            signal,
            sample_rate,
            block_size=1024,  # Smaller block size for faster processing
            hop_size=256,  # Higher time resolution
            min_freq=60,  # Wider frequency range
            max_freq=1000,
            confidence_threshold=0.15,  # Lower threshold for more detections
        )

        print(f"\nAlternative analysis with different parameters:")
        print(f"Block size: 1024, Hop size: 256, Freq range: 60-1000 Hz")
        print(f"Voiced frames: {np.sum(frequencies > 0)}/{len(frequencies)}")
        if np.sum(frequencies > 0) > 0:
            voiced_freqs = frequencies[frequencies > 0]
            print(
                f"Frequency range: {np.min(voiced_freqs):.1f} - {np.max(voiced_freqs):.1f} Hz"
            )

    except Exception as e:
        print(f"Manual analysis failed: {e}")

    print("\n" + "=" * 60)
    print("🎯 USAGE INSTRUCTIONS")
    print("=" * 60)
    print("To analyze any audio file, simply use:")
    print("results = analyze_audio_pitch('your_audio_file.wav')")
    print("\nSupported formats: WAV, MP3, FLAC, M4A, and more (via librosa)")
    print("Adjust parameters as needed for your specific audio content.")
    print("=" * 60)
