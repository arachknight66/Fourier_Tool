from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class SignalConfig:
    """
    Configuration for signal properties.
    """
    sampling_rate: float
    duration: Optional[float] = None
    n_samples: Optional[int] = None

@dataclass
class FilterConfig:
    """
    Configuration for filter design[cite: 40, 58].
    """
    filter_type: str  # 'lowpass', 'highpass', 'bandpass', 'notch'
    cutoff_freq: float
    bandwidth: Optional[float] = None  # For bandpass/notch
    order: int = 4

@dataclass
class ProcessingConfig:
    """
    Global processing configuration.
    """
    window_type: str = 'hann'   
    normalize: bool = True
    detrend: bool = True
    zero_padding_factor: int = 1
