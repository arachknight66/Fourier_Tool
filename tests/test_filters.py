import unittest
import numpy as np
from src.filters.lowpass import LowPassFilter
from src.filters.highpass import HighPassFilter
from src.filters.notch import NotchFilter

class TestFilters(unittest.TestCase):
    def setUp(self):
        self.freqs = np.linspace(0, 100, 200) # 0 to 100 Hz
        self.ones = np.ones_like(self.freqs, dtype=complex) # Flat spectrum

    def test_lowpass_ideal(self):
        """Test if ideal lowpass zeroes out frequencies above cutoff."""
        cutoff = 30
        lp = LowPassFilter(cutoff_freq=cutoff, order=None) # Ideal
        
        mask = lp.compute_mask(self.freqs)
        
        # Check a frequency below cutoff (e.g., 10Hz) -> Should be 1.0
        self.assertEqual(mask[20], 1.0) 
        # Check a frequency above cutoff (e.g., 80Hz) -> Should be 0.0
        self.assertEqual(mask[160], 0.0)

    def test_highpass_butterworth(self):
        """Test if butterworth highpass attenuates low frequencies."""
        cutoff = 50
        hp = HighPassFilter(cutoff_freq=cutoff, order=2)
        
        mask = hp.compute_mask(self.freqs)
        
        # Freq below cutoff (10Hz) should be attenuated (< 1.0)
        self.assertLess(mask[20], 0.1)
        # Freq way above cutoff (90Hz) should be passed (~ 1.0)
        self.assertGreater(mask[180], 0.9)

    def test_notch_filter(self):
        """Test if notch filter removes the specific target frequency."""
        center = 60
        bw = 4
        notch = NotchFilter(center_freq=center, bandwidth=bw)
        
        mask = notch.compute_mask(self.freqs)
        
        # Find index closest to 60Hz
        idx_60 = (np.abs(self.freqs - 60)).argmin()
        
        # Should be 0.0 at center
        self.assertEqual(mask[idx_60], 0.0)
        # Should be 1.0 far away (e.g., 20Hz)
        self.assertEqual(mask[40], 1.0)