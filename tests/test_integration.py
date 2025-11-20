import unittest
import numpy as np
from src.core.pipeline import SignalPipeline
from src.filters.lowpass import LowPassFilter
from src.utils.signal_generator import SignalGenerator

class TestIntegration(unittest.TestCase):
    def test_full_denoising_pipeline(self):
        fs = 1000
        gen = SignalGenerator(fs, duration=1.0)
        
        # Create signal: 5Hz (Signal) + 100Hz (Noise)
        clean = gen.sine_wave(5)
        noise = gen.sine_wave(100, amplitude=0.5)
        raw = clean + noise
        
        # Setup Pipeline
        pipeline = SignalPipeline(fs)
        
        # Create Filter: Lowpass at 20Hz (Should keep 5Hz, remove 100Hz)
        lp_filter = LowPassFilter(cutoff_freq=20)
        
        # Run
        results = pipeline.run(raw, filter_object=lp_filter)
        output = results["output_signal"]
        
        # Verification:
        # 1. Output should closely match 'clean' signal (MSE should be low)
        mse = np.mean((clean - output)**2)
        self.assertLess(mse, 0.1) # Allow some error due to filter transition
        
        # 2. Signal energy should be preserved, Noise energy removed
        # (Simple check: output variance should be closer to clean variance than raw variance)
        self.assertTrue(abs(np.var(output) - np.var(clean)) < abs(np.var(output) - np.var(raw)))