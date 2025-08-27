#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MUSI6201 Audio Blocking Assignment

In this assignment, you will convert your pseudo code activity into actual
code in order to implement an audio "blocking" function.

Your function may use existing python libraries to load the audio (e.g., scipy,
librosa)but the rest will be done with just numpy.

Please read the parameters carefully and use the class discussion on Canvas if
you have questions.

Please note the "pad" condition (i.e., padding with extra zeros so that the
total number of samples are an integer multiple of the hop size).
If you did not think of this in class, try working it out in pseudo code first.

Parameters
----------
audio_input : np.ndarray or str
    Audio input. Should be able to come from either:
    - A NumPy array containing the audio signal.
    - A string path to an audio file (e.g., 'audio.wav').
sr : int
    Sampling rate. Required if input is np.ndarray
frame_size : int, optional
    Size of each frame in samples (make default 2048).
hop_ratio : float, optional
    Hop size as a ratio of the frame_size (make default .5).
pad : bool, optional
    If True (default), pads the signal with zeros to ensure all frames are the same length.
    If False, discards the last incomplete frame.

Returns
-------
frames : np.ndarray
    2D array of shape (n_frames, frame_size).
times : np.ndarray
    Array of start times (in seconds) for each frame.
"""

import numpy as np
import scipy.io.wavfile as wav


def audio_blocking(audio_input, sr=None, frame_size=2048, hop_ratio=0.5, pad=True):

    sampling_rate, audio_input = wav.read(audio_input)
    if sr is not None:
        sampling_rate = sr

    if audio_input.ndim == 1:
        audio_signal = audio_input
    else:
        audio_signal = audio_input[:, 0]

    hop_size = int(frame_size * hop_ratio)

    signal_length = len(audio_signal)

    # Padded length should be such that it is an integer multiple of the hop size
    # Lets say the total number fo samples is 15, frame size is 4 and hop size is 2
    # Then the number of hops is 6 = (15 - 4) / 2 + 1
    # There are 6 frames in total but last frame is incomplete (3 samples 13, 14, 15 are left unframed)

    n_hops = (signal_length - frame_size) // hop_size + 1

    if pad:
        if (signal_length - frame_size) % hop_size != 0:  # padding is required
            n_hops = (signal_length - frame_size) // hop_size + 1
            signal_length_with_padding = n_hops * hop_size + frame_size
            required_padding = signal_length_with_padding - signal_length
            audio_signal = np.pad(audio_signal, (0, required_padding), mode="constant")
            n_frames = (signal_length - frame_size) // hop_size + 1
    else:
        if (signal_length - frame_size) % hop_size != 0:
            n_hops = (signal_length - frame_size) // hop_size
            signal_length_without_padding = n_hops * hop_size + frame_size
            audio_signal = audio_signal[
                :signal_length_without_padding
            ]  # discard the last incomplete frame
            n_frames = (signal_length - frame_size) // hop_size + 1

    frames = np.zeros((n_frames, frame_size))

    for i in range(n_frames):
        start_index = i * hop_size
        end_index = start_index + frame_size
        frames[i, :] = audio_signal[start_index:end_index]

    times = np.arange(n_frames) * hop_size / sampling_rate

    return frames, times


if __name__ == "__main__":
    audio_file = "prelude_cmaj_10s.wav"

    frames, times = audio_blocking(audio_file, frame_size=1024, hop_ratio=0.5, pad=True)

    print(f"Results:")
    print(f"   - Frames shape: {frames.shape}")
    print(f"   - Times shape: {times.shape}")
    print(f"   - Number of frames: {frames.shape[0]}")
    print(f"   - Frame size: {frames.shape[1]} samples")
    print(f"   - Expected dimensions from comment: (857, 1024)")
