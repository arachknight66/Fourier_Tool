import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.pipeline import SignalPipeline
from src.filters.adaptive import SpectralSubtractionFilter
from src.utils.signal_generator import SignalGenerator

def run_audio_demo():
    print("Running Audio Denoising Demo...")
    
    # 1. Generate "Audio-like" data
    fs = 44100.0
    gen = SignalGenerator(fs, duration=0.1) # Short clip for visualization
    
    # Simulate a chord (Speech/Music)
    chord = gen.sine_wave(440) + gen.sine_wave(554) + gen.sine_wave(659)
    # Add broadband white noise (background hiss)
    noise = gen.white_noise(amplitude=0.8)
    raw_audio = chord + noise
    
    # 2. Apply Spectral Subtraction
    # This method estimates the noise floor and subtracts it from the magnitude spectrum
    pipeline = SignalPipeline(fs)
    adaptive_filter = SpectralSubtractionFilter(noise_floor_factor=1.5)
    
    res = pipeline.run(raw_audio, filter_object=adaptive_filter)
    cleaned_audio = res["output_signal"]
    
    # 3. Plot Spectrograms to show background noise removal
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax1.specgram(raw_audio, Fs=fs, NFFT=1024, noverlap=512, cmap='inferno')
    ax1.set_title("Spectrogram: Noisy Audio")
    ax1.set_ylabel("Frequency (Hz)")
    
    ax2.specgram(cleaned_audio, Fs=fs, NFFT=1024, noverlap=512, cmap='viridis')
    ax2.set_title("Spectrogram: Denoised Audio (Adaptive)")
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_xlabel("Time (s)")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_audio_demo()