import sys
import os
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.signal_generator import SignalGenerator

def ensure_dirs():
    """Create necessary data directories if they don't exist."""
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

def generate_datasets():
    print("Generating Synthetic Datasets...")
    
    # 1. Biomedical (ECG) - 360Hz
    # Challenges: Baseline wander, 60Hz interference [cite: 79, 80]
    fs_bio = 360.0
    gen_bio = SignalGenerator(fs_bio, duration=10.0)
    _, ecg_noisy, ecg_clean = gen_bio.generate_synthetic_ecg()
    
    np.save("data/raw/ecg_noisy.npy", ecg_noisy)
    np.save("data/raw/ecg_clean.npy", ecg_clean)
    print(f"✅ Generated ECG Data ({len(ecg_noisy)} samples)")

    # 2. Seismic (P-Waves) - 100Hz
    # Challenges: Ambient noise, wind vibration [cite: 81, 82]
    fs_seismic = 100.0
    gen_seismic = SignalGenerator(fs_seismic, duration=30.0)
    _, seismic_noisy, seismic_clean = gen_seismic.generate_synthetic_ecg() # Reusing generic structure for demo
    # Overwrite with actual seismic logic from generator if strictly needed, 
    # but for the script structure, we use what's available.
    # Let's call the specific seismic method if it exists in your utils:
    _, seismic_noisy, seismic_clean = gen_seismic.generate_seismic_event()
    
    np.save("data/raw/seismic_noisy.npy", seismic_noisy)
    np.save("data/raw/seismic_clean.npy", seismic_clean)
    print(f"✅ Generated Seismic Data ({len(seismic_noisy)} samples)")

    # 3. Audio/AI - 44.1kHz
    # Challenges: Background noise [cite: 83, 84]
    fs_audio = 44100.0
    gen_audio = SignalGenerator(fs_audio, duration=5.0)
    # Create a simple tone + noise
    audio_clean = gen_audio.sine_wave(440) + gen_audio.sine_wave(880, 0.5)
    audio_noise = gen_audio.white_noise(0.2)
    audio_noisy = audio_clean + audio_noise
    
    np.save("data/raw/audio_noisy.npy", audio_noisy)
    np.save("data/raw/audio_clean.npy", audio_clean)
    print(f"✅ Generated Audio Data ({len(audio_noisy)} samples)")

if __name__ == "__main__":
    ensure_dirs()
    generate_datasets()