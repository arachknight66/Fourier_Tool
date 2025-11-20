import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.pipeline import SignalPipeline
from src.filters.highpass import HighPassFilter
from src.filters.notch import NotchFilter
from src.utils.signal_generator import SignalGenerator

def run_ecg_demo():
    print("Running ECG Filtering Demo...")
    
    # 1. Generate Synthetic ECG
    fs = 360.0
    gen = SignalGenerator(fs, duration=5.0)
    t, raw_ecg, ideal_ecg = gen.generate_synthetic_ecg()
    
    # 2. Define Pipeline
    pipeline = SignalPipeline(fs)
    
    # 3. Step 1: Remove Baseline Wander (Drift)
    # High-pass filter > 0.5 Hz
    hp = HighPassFilter(cutoff_freq=0.5, order=2)
    res_step1 = pipeline.run(raw_ecg, filter_object=hp)
    ecg_no_drift = res_step1["output_signal"]
    
    # 4. Step 2: Remove Power Line Hum (60Hz)
    # Notch filter at 60Hz
    notch = NotchFilter(center_freq=60, bandwidth=4.0)
    res_step2 = pipeline.run(ecg_no_drift, filter_object=notch)
    ecg_clean = res_step2["output_signal"]
    
    # 5. Visualize the progression
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    
    # Plot Raw
    axes[0].plot(t[:1000], raw_ecg[:1000], color='red', alpha=0.7)
    axes[0].set_title("1. Raw ECG (Wander + 60Hz Noise)")
    
    # Plot after High-pass
    axes[1].plot(t[:1000], ecg_no_drift[:1000], color='orange', alpha=0.8)
    axes[1].set_title("2. After High-Pass (Wander Removed)")
    
    # Plot Final
    axes[2].plot(t[:1000], ecg_clean[:1000], color='green')
    axes[2].plot(t[:1000], ideal_ecg[:1000], 'k--', alpha=0.4, label='Ideal')
    axes[2].set_title("3. Final Output (Notch Filtered)")
    axes[2].legend()
    
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_ecg_demo()