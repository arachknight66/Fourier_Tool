import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.pipeline import SignalPipeline
from src.filters.lowpass import LowPassFilter
from src.filters.bandpass import BandPassFilter
from src.filters.notch import NotchFilter

def evaluate_all():
    Path("results/metrics").mkdir(parents=True, exist_ok=True)
    results = []

    # --- 1. Evaluate ECG ---
    print("Evaluating Biomedical (ECG)...")
    raw = np.load("data/raw/ecg_noisy.npy")
    clean = np.load("data/raw/ecg_clean.npy")
    fs = 360.0
    
    pipeline = SignalPipeline(fs)
    # Use the filters identified in notebook 04: Notch 60Hz
    filt = NotchFilter(center_freq=60, bandwidth=5)
    
    res = pipeline.run(raw, filter_object=filt)
    out = res["output_signal"]
    
    snr_in = pipeline.reconstructor.calculate_snr(clean, raw - clean)
    snr_out = pipeline.reconstructor.calculate_snr(clean, out - clean)
    
    results.append({
        "Domain": "Biomedical", 
        "Filter": "Notch(60Hz)", 
        "Input_SNR": snr_in, 
        "Output_SNR": snr_out,
        "Gain_dB": snr_out - snr_in
    })

    # --- 2. Evaluate Seismic ---
    print("Evaluating Seismic...")
    raw = np.load("data/raw/seismic_noisy.npy")
    clean = np.load("data/raw/seismic_clean.npy")
    fs = 100.0
    
    pipeline = SignalPipeline(fs)
    # Bandpass 2-10Hz
    filt = BandPassFilter(2, 10)
    
    res = pipeline.run(raw, filter_object=filt)
    out = res["output_signal"]
    
    snr_in = pipeline.reconstructor.calculate_snr(clean, raw - clean)
    snr_out = pipeline.reconstructor.calculate_snr(clean, out - clean)
    
    results.append({
        "Domain": "Seismic", 
        "Filter": "Bandpass(2-10Hz)", 
        "Input_SNR": snr_in, 
        "Output_SNR": snr_out,
        "Gain_dB": snr_out - snr_in
    })
    
    # Save Results
    df = pd.DataFrame(results)
    df.to_csv("results/metrics/performance_summary.csv", index=False)
    print("✅ Evaluation Complete. Results saved.")
    print(df)

if __name__ == "__main__":
    try:
        evaluate_all()
    except FileNotFoundError:
        print("❌ Data missing. Run 'prepare_data.py' first.")