import numpy as np
from base_filter import *

class SpectralSubtractionFilter(BaseFilter):
    def __init__(self, noise_floor_factor: float = 1.5):
        """
        Adaptive filter that subtracts estimated noise from the signal.
        
        Args:
            noise_floor_factor: Multiplier for the noise threshold. 
                                Higher = more aggressive denoising.
        """
        super().__init__()
        self.alpha = noise_floor_factor

    def compute_mask(self, frequencies: np.ndarray) -> np.ndarray:
        # Note: This filter requires the spectrum magnitude to compute the mask,
        # so the logic is slightly different. We override apply() instead.
        return np.ones_like(frequencies) 

    def apply(self, frequencies: np.ndarray, complex_spectrum: np.ndarray) -> np.ndarray:
        """
        Overrides base apply to perform magnitude-based subtraction.
        """
        magnitude = np.abs(complex_spectrum)
        phase = np.angle(complex_spectrum)
        
        # 1. Estimate Noise Profile
        # Assumption: The lowest 10% of magnitudes represent the noise floor 
        # (simple heuristic for this project level)
        noise_estimate = np.percentile(magnitude, 10)
        
        # 2. Spectral Subtraction
        # New Magnitude = |S(f)| - (alpha * Noise)
        new_magnitude = magnitude - (self.alpha * noise_estimate)
        
        # 3. Half-wave rectification (prevent negative magnitude)
        new_magnitude = np.maximum(new_magnitude, 0)
        
        # 4. Reconstruct Complex Spectrum
        # S_new(f) = |S_new| * e^(j * phase)
        filtered_spectrum = new_magnitude * np.exp(1j * phase)
        
        # Save state for consistency
        self.state = FilterState(mask=new_magnitude / (magnitude + 1e-10))
        
        return filtered_spectrum