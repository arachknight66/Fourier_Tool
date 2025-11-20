import numpy as np
from .base_filter import BaseFilter

class BandPassFilter(BaseFilter):
    def __init__(self, low_cutoff: float, high_cutoff: float):
        """
        Args:
            low_cutoff: Lower frequency bound.
            high_cutoff: Upper frequency bound.
        """
        super().__init__()
        self.low = low_cutoff
        self.high = high_cutoff

    def compute_mask(self, frequencies: np.ndarray) -> np.ndarray:
        abs_freqs = np.abs(frequencies)
        
        # Ideal Bandpass: 1.0 inside the range, 0.0 outside
        mask = np.where(
            (abs_freqs >= self.low) & (abs_freqs <= self.high), 
            1.0, 
            0.0
        )
        return mask