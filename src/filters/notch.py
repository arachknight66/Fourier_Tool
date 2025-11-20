import numpy as np
from base_filter import *

class NotchFilter(BaseFilter):
    def __init__(self, center_freq: float, bandwidth: float):
        """
        Args:
            center_freq: The frequency to reject (e.g., 60Hz).
            bandwidth: The width of the rejection band (e.g., 2Hz).
        """
        super().__init__()
        self.center = center_freq
        self.bw = bandwidth

    def compute_mask(self, frequencies: np.ndarray) -> np.ndarray:
        abs_freqs = np.abs(frequencies)
        
        # Reject frequencies within [center - bw/2, center + bw/2]
        lower_bound = self.center - (self.bw / 2)
        upper_bound = self.center + (self.bw / 2)
        
        # Invert logic: Pass everything EXCEPT the target band
        mask = np.where(
            (abs_freqs >= lower_bound) & (abs_freqs <= upper_bound),
            0.0,
            1.0
        )
        return mask