from scipy.io import wavfile
import numpy as np
import math

def block_audio(audio_input, sr=None, frame_size=2048, hop_ratio=0.5, pad=True):
    """
    Parameters
    ----------
    audio_input : np.ndarray or str
        Audio input. Should be able to come from either:
        - A NumPy array containing the audio signal.
        - A string path to an audio file (e.g., 'audio.wav').
    sr : int
        Sampling rate. Required if input is np.ndarray
    frame_size : int, optional
        Size of each frame in samples (default 2048).
    hop_ratio : float, optional
        Hop size as a ratio of the frame_size (default 0.5).
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
    
    # Handle input type
    if isinstance(audio_input, str):
        sr, audio_input = wavfile.read(audio_input)
    elif sr is None:
        raise ValueError("Must provide sampling rate with numpy array or file path as a string")
    
    # Convert to float in range [-1, 1)
    if audio_input.dtype == np.float32:
        audio_input = audio_input
    else:
        # Determine bit depth and convert
        if audio_input.dtype == np.uint8:
            nbits = 8
        elif audio_input.dtype == np.int16:
            nbits = 16
        elif audio_input.dtype == np.int32:
            nbits = 32
        else:
            raise ValueError(f"Unsupported audio dtype: {audio_input.dtype}")
        
        audio_input = audio_input / float(2**(nbits - 1))
    
    # Convert to mono if stereo
    if len(audio_input.shape) > 1:
        audio_input = np.mean(audio_input, axis=1)
    
    # Calculate hop size as integer
    hop_size = int(hop_ratio * frame_size)
    
    # Calculate number of frames
    if pad:
        # Include all possible frames, padding the last one if necessary
        num_blocks = math.ceil((len(audio_input) - frame_size) / hop_size) + 1
        # Ensure we have at least one block even for very short audio
        num_blocks = max(1, num_blocks)
    else:
        # Only include complete frames
        num_blocks = max(0, (len(audio_input) - frame_size) // hop_size + 1)
    
    # Initialize output arrays
    audio_blocks = np.zeros([num_blocks, frame_size])
    
    # Compute time stamps
    times = (np.arange(0, num_blocks) * hop_size) / sr
    
    # Extract frames
    for n in range(num_blocks):
        i_start = n * hop_size
        i_stop = i_start + frame_size
        
        if i_stop <= len(audio_input):
            # Complete frame
            audio_blocks[n] = audio_input[i_start:i_stop]
        else:
            # Incomplete frame (only happens when pad=True)
            remaining_samples = len(audio_input) - i_start
            if remaining_samples > 0:
                audio_blocks[n, :remaining_samples] = audio_input[i_start:]
                # Rest of the frame is already zeros from initialization
    
    return audio_blocks, times