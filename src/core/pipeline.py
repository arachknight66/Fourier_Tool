import numpy as np
from .preprocessing import Preprocessor
from .fourier_transform import FourierTransformer
from .reconstruction import Reconstructor
from .config import ProcessingConfig

class SignalPipeline:
    def __init__(self, sampling_rate: float, config: ProcessingConfig = None):
        self.fs = sampling_rate
        self.config = config or ProcessingConfig()
        
        self.preprocessor = Preprocessor(self.config)
        self.transformer = FourierTransformer(self.fs)
        self.reconstructor = Reconstructor(self.fs)

    def run(self, raw_signal: np.ndarray, filter_object=None):
        """
        Executes the full Signal Processing Pipeline:
        Preprocessing -> FFT -> Filtering -> IFFT
        """
        # 1. Preprocessing 
        clean_signal = self.preprocessor.process(raw_signal)
        
        # 2. Spectral Transformation [cite: 57]
        spectrum = self.transformer.compute_fft(clean_signal)
        
        # 3. Apply Filter (if provided) [cite: 46, 47]
        # Y(f) = S(f) * H(f)
        filtered_spectrum = spectrum.complex_spectrum
        if filter_object:
            filtered_spectrum = filter_object.apply(
                spectrum.frequencies, 
                spectrum.complex_spectrum
            )
            
        # 4. Reconstruction 
        reconstructed_signal = self.reconstructor.inverse_transform(filtered_spectrum)
        
        return {
            "original_spectrum": spectrum,
            "filtered_spectrum_data": filtered_spectrum,
            "output_signal": reconstructed_signal
        }
