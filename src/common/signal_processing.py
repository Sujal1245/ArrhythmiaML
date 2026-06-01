import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_filter(signal, fs=360, lo=0.5, hi=40.0, order=4):
    """Apply a butterworth bandpass filter to the signal."""
    nyq = fs / 2.0
    b, a = butter(order, [lo / nyq, hi / nyq], btype='band')
    return filtfilt(b, a, signal)

def normalize_segment(segment):
    """Z-score normalize a signal segment."""
    return (segment - np.mean(segment)) / (np.std(segment) + 1e-8)
