import numpy as np
from .base_filter import BaseFilter

class LowPassFilter(BaseFilter):
    def __init__(self, cutoff_freq: float, order: int = None):
        """
        Args:
            cutoff_freq: Frequency in Hz above which signals are attenuated.
            order: If None, uses Ideal (Brick-wall) filter. 
                   If integer > 0, simulates Butterworth roll-off.
        """
        super().__init__()
        self.cutoff = cutoff_freq
        self.order = order

    def compute_mask(self, frequencies: np.ndarray) -> np.ndarray:
        abs_freqs = np.abs(frequencies)
        
        if self.order is None:
            # Ideal (Brick-wall) Filter
            mask = np.where(abs_freqs <= self.cutoff, 1.0, 0.0)
        else:
            # Butterworth Filter Response: 1 / sqrt(1 + (f/fc)^2n)
            # Note: We use squared magnitude for the mask on complex numbers
            mask = 1 / np.sqrt(1 + (abs_freqs / self.cutoff) ** (2 * self.order))
            
        return mask
    