import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.pipeline import SignalPipeline
from src.filters.lowpass import LowPassFilter
from src.utils.signal_generator import SignalGenerator

def run_basic_demo():
    print("Running Basic Filtering Demo...")
    
    # 1. Setup: Define signal parameters
    fs = 1000.0
    duration = 2.0
    gen = SignalGenerator(fs, duration)
    
    # 2. Generate Data: Clean Sine (5Hz) + High Freq Noise (50Hz)
    t = gen.t
    clean = gen.sine_wave(freq=5, amplitude=1.0)
    noise = gen.sine_wave(freq=50, amplitude=0.4) 
    raw = clean + noise
    
    # 3. Configure Pipeline
    # We want to keep 5Hz and remove 50Hz. A cutoff of 20Hz is safe.
    pipeline = SignalPipeline(fs)
    lp_filter = LowPassFilter(cutoff_freq=20, order=4)
    
    # 4. Run
    results = pipeline.run(raw, filter_object=lp_filter)
    output = results["output_signal"]
    
    # 5. Visualize
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t[:200], raw[:200], 'r-', alpha=0.6, label='Noisy Input (5Hz + 50Hz)')
    plt.plot(t[:200], clean[:200], 'k--', label='Ideal Clean Signal')
    plt.title("Input Signal")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(t[:200], output[:200], 'g-', linewidth=2, label='Filtered Output')
    plt.title("Output Signal (Low-Pass Filtered)")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_basic_demo()