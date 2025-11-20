import numpy as np
from scipy.fft import irfft, ifft
from scipy.signal import welch

class Reconstructor:
    def __init__(self, sampling_rate: float):
        self.fs = sampling_rate

    def inverse_transform(self, complex_spectrum: np.ndarray, real_output: bool = True) -> np.ndarray:
        """
        Reconstructs the time-domain signal using Inverse Fourier Transform[cite: 49, 50].
        """
        if real_output:
            return irfft(complex_spectrum)
        return ifft(complex_spectrum)

    def calculate_mse(self, original: np.ndarray, denoised: np.ndarray) -> float:
        """
        Calculates Mean Squared Error[cite: 18].
        """
        # Ensure lengths match (cropping might be needed due to padding)
        min_len = min(len(original), len(denoised))
        return np.mean((original[:min_len] - denoised[:min_len]) ** 2)

    def calculate_snr(self, original: np.ndarray, noise: np.ndarray) -> float:
        """
        Calculates Signal-to-Noise Ratio in dB[cite: 18].
        """
        signal_power = np.mean(original ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power == 0:
            return np.inf
            
        return 10 * np.log10(signal_power / noise_power)

    def compute_psd(self, data: np.ndarray) -> tuple:
        """
        Computes Power Spectral Density[cite: 19, 59].
        """
        freqs, psd = welch(data, self.fs)
        return freqs, psd
