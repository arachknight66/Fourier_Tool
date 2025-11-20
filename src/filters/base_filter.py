import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class FilterState:
    """Stores the mask used for the last filtering operation for visualization."""
    mask: np.ndarray

class BaseFilter(ABC):
    """
    Abstract base class for all frequency-domain filters.
    """
    def __init__(self):
        self.state = None

    @abstractmethod
    def compute_mask(self, frequencies: np.ndarray) -> np.ndarray:
        """
        Generates the frequency domain mask (H(f)).
        Must be implemented by child classes.
        """
        pass

    def apply(self, frequencies: np.ndarray, complex_spectrum: np.ndarray) -> np.ndarray:
        """
        Applies the filter: Y(f) = S(f) * H(f)
        """
        # Generate the mask (Transfer Function H(f))
        mask = self.compute_mask(frequencies)
        
        # Store state for visualization/debugging
        self.state = FilterState(mask=mask)
        
        # Apply element-wise multiplication
        return complex_spectrum * mask