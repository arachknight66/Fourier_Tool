import numpy as np
from scipy.fft import fft, fftfreq, rfft, rfftfreq
from dataclasses import dataclass

@dataclass
class Spectrum:
    frequencies: np.ndarray
    magnitude: np.ndarray
    phase: np.ndarray
    complex_spectrum: np.ndarray

class FourierTransformer:
    def __init__(self, sampling_rate: float):
        self.fs = sampling_rate

    def compute_fft(self, signal_data: np.ndarray, real_only: bool = True) -> Spectrum:
        """
        Computes the Discrete Fourier Transform.
        """
        n = len(signal_data)
        
        if real_only:
            # rfft is more efficient for real-valued signals (common in sensors)
            complex_spectrum = rfft(signal_data)
            frequencies = rfftfreq(n, d=1/self.fs)
        else:
            complex_spectrum = fft(signal_data)
            frequencies = fftfreq(n, d=1/self.fs)

        # Compute Magnitude and Phase [cite: 57]
        magnitude = np.abs(complex_spectrum)
        phase = np.angle(complex_spectrum)

        return Spectrum(
            frequencies=frequencies,
            magnitude=magnitude,
            phase=phase,
            complex_spectrum=complex_spectrum
        )
