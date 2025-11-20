import numpy as np

class SignalGenerator:
    def __init__(self, sampling_rate: float, duration: float):
        self.fs = sampling_rate
        self.duration = duration
        self.t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

    def sine_wave(self, freq: float, amplitude: float = 1.0) -> np.ndarray:
        """Generates a pure sine wave."""
        return amplitude * np.sin(2 * np.pi * freq * self.t)

    def white_noise(self, amplitude: float = 1.0) -> np.ndarray:
        """Generates Gaussian white noise."""
        return np.random.normal(0, amplitude, len(self.t))

    def generate_synthetic_ecg(self) -> tuple:
        """
        Simulates a biomedical signal with defects mentioned in the synopsis:
        1. QRS Complex (Signal)
        2. 60Hz Power line interference (Noise) [cite: 80]
        3. Baseline wander (Low freq noise) [cite: 80]
        """
        # 1. Mock QRS complex (periodic pulses)
        # Using a simple periodic sinc/gaussian approximation for demonstration
        heart_rate_hz = 1.2  # ~72 BPM
        signal = 0.8 * np.sin(2 * np.pi * heart_rate_hz * self.t) ** 14 
        
        # 2. 60Hz Interference
        interference = self.sine_wave(freq=60, amplitude=0.15)
        
        # 3. Baseline Wander (0.5 Hz)
        wander = self.sine_wave(freq=0.5, amplitude=0.2)
        
        raw_signal = signal + interference + wander + self.white_noise(0.05)
        return self.t, raw_signal, signal # Return time, noisy, clean

    def generate_seismic_event(self) -> tuple:
        """
        Simulates a seismic P-wave buried in noise[cite: 82].
        """
        # P-wave approximated as a Ricker wavelet or damped sine
        center_time = self.duration / 2
        f0 = 5.0 # dominant frequency Hz
        
        # Ricker wavelet formula
        t_shifted = self.t - center_time
        p_wave = (1 - 2 * (np.pi * f0 * t_shifted)**2) * np.exp(-(np.pi * f0 * t_shifted)**2)
        
        # Heavy ambient noise (Wind/Vibration)
        noise = self.white_noise(amplitude=0.6)
        
        return self.t, p_wave + noise, p_wave