import numpy as np
from .base_filter import BaseFilter

class HighPassFilter(BaseFilter):
    def __init__(self, cutoff_freq: float, order: int = None):
        """
        Args:
            cutoff_freq: Frequency in Hz below which signals are attenuated.
        """
        super().__init__()
        self.cutoff = cutoff_freq
        self.order = order

    def compute_mask(self, frequencies: np.ndarray) -> np.ndarray:
        abs_freqs = np.abs(frequencies)
        
        # Avoid division by zero for DC component in Butterworth
        with np.errstate(divide='ignore'):
            if self.order is None:
                # Ideal Filter
                mask = np.where(abs_freqs >= self.cutoff, 1.0, 0.0)
            else:
                # Butterworth High-pass
                mask = 1 / np.sqrt(1 + (self.cutoff / abs_freqs) ** (2 * self.order))
                # Fix DC component (freq=0) which goes to 0
                mask[np.isnan(mask)] = 0.0
                
        return mask