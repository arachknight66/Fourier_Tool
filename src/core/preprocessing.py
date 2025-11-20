import numpy as np
from scipy import signal
from typing import Tuple, Optional

class Preprocessor:
    def __init__(self, config):
        self.config = config

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalizes signal amplitude to range [-1, 1] or Z-score standardization.
        """
        if np.std(data) == 0:
            return data
        return (data - np.mean(data)) / np.std(data)

    def remove_trend(self, data: np.ndarray) -> np.ndarray:
        """
        Removes linear trend from the signal to improve FFT accuracy.
        """
        return signal.detrend(data)

    def apply_window(self, data: np.ndarray, window_type: str = 'hann') -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies a window function to reduce spectral leakage.
        Returns:
            tuple: (windowed_data, window_weights)
        """
        # Create window based on signal length
        if window_type == 'hann':
            win = np.hanning(len(data))
        elif window_type == 'hamming':
            win = np.hamming(len(data))
        elif window_type == 'blackman':
            win = np.blackman(len(data))
        else:
            win = np.ones(len(data))
            
        return data * win, win

    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Execute full preprocessing pipeline.
        """
        processed = data.copy()
        
        if self.config.detrend:
            processed = self.remove_trend(processed)
            
        if self.config.normalize:
            processed = self.normalize(processed)
            
        # Note: Windowing is usually applied right before FFT, 
        # but can be done here if strictly required by pipeline.
        return processed
