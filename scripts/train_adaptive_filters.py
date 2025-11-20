import sys
import os
import numpy as np
import json
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.pipeline import SignalPipeline
from src.filters.adaptive import SpectralSubtractionFilter

def calibrate_audio_filter():
    print("Calibrating Adaptive Filter for Audio...")
    
    # Load Data
    try:
        noisy = np.load("data/raw/audio_noisy.npy")
        clean = np.load("data/raw/audio_clean.npy")
    except FileNotFoundError:
        print("❌ Data not found. Run 'prepare_data.py' first.")
        return

    fs = 44100.0
    pipeline = SignalPipeline(fs)
    
    # Grid Search for optimal 'alpha' (Noise Floor Factor)
    alphas = np.linspace(0.5, 3.0, 10)
    best_mse = float('inf')
    best_alpha = 1.0
    
    print(f"Testing alpha values: {alphas}")
    
    for alpha in alphas:
        # Setup Filter
        filt = SpectralSubtractionFilter(noise_floor_factor=alpha)
        
        # Run Pipeline (use a subset for speed)
        res = pipeline.run(noisy[:10000], filter_object=filt)
        output = res["output_signal"]
        
        # Calculate MSE [cite: 18, 41]
        mse = np.mean((clean[:10000] - output)**2)
        
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
            
    print(f"✅ Best Alpha found: {best_alpha:.2f} (MSE: {best_mse:.6f})")
    
    # Save Model Config
    Path("results/models").mkdir(parents=True, exist_ok=True)
    config = {"filter_type": "spectral_subtraction", "best_alpha": best_alpha}
    
    with open("results/models/audio_calibration.json", "w") as f:
        json.dump(config, f, indent=4)
    print("Saved calibration to 'results/models/audio_calibration.json'")

if __name__ == "__main__":
    calibrate_audio_filter()