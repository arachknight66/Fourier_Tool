import unittest
import numpy as np
from src.core.preprocessing import Preprocessor
from src.core.fourier_transform import FourierTransformer
from src.core.reconstruction import Reconstructor
from src.core.config import ProcessingConfig

class TestCore(unittest.TestCase):
    def setUp(self):
        self.fs = 1000.0
        self.n_samples = 1000
        self.t = np.linspace(0, 1, self.n_samples, endpoint=False)
        self.sine_10hz = np.sin(2 * np.pi * 10 * self.t)
        self.config = ProcessingConfig(detrend=False, normalize=False)

    def test_preprocessing_normalization(self):
        prep = Preprocessor(self.config)
        data = np.array([10, 20, 30])
        norm_data = prep.normalize(data)
        self.assertTrue(np.isclose(np.mean(norm_data), 0, atol=1e-7))
        self.assertTrue(np.isclose(np.std(norm_data), 1, atol=1e-7))

    def test_fft_peak_detection(self):
        """Checks if FFT correctly identifies the 10Hz frequency component."""
        transformer = FourierTransformer(self.fs)
        spectrum = transformer.compute_fft(self.sine_10hz)
        
        # Find frequency with max magnitude
        peak_idx = np.argmax(spectrum.magnitude)
        peak_freq = spectrum.frequencies[peak_idx]
        
        self.assertTrue(np.isclose(peak_freq, 10.0, atol=1.0))

    def test_reconstruction_integrity(self):
        """Checks Parseval's theorem: IFFT(FFT(Signal)) == Signal."""
        transformer = FourierTransformer(self.fs)
        reconstructor = Reconstructor(self.fs)
        
        spectrum = transformer.compute_fft(self.sine_10hz)
        reconstructed = reconstructor.inverse_transform(spectrum.complex_spectrum)
        
        # Check MSE is near zero (ignoring tiny floating point errors)
        mse = reconstructor.calculate_mse(self.sine_10hz, reconstructed)
        self.assertLess(mse, 1e-10)