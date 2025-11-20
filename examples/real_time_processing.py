import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.pipeline import SignalPipeline
from src.filters.lowpass import LowPassFilter
from src.utils.signal_generator import SignalGenerator

def simulate_streaming_data(total_duration, chunk_size, fs):
    """Generator that yields chunks of data."""
    gen = SignalGenerator(fs, total_duration)
    # Create a long signal: varying frequency sine wave
    t = gen.t
    clean = np.sin(2 * np.pi * (1 + 0.5*t) * t) # Chirp
    noise = gen.white_noise(0.3)
    full_signal = clean + noise
    
    total_samples = len(full_signal)
    cursor = 0
    
    while cursor < total_samples:
        end = min(cursor + chunk_size, total_samples)
        yield full_signal[cursor:end]
        cursor += chunk_size

def run_realtime_demo():
    print("Running Real-Time Processing Simulation...")
    
    fs = 1000.0
    chunk_size = 100 # Process 100 samples (0.1s) at a time
    duration = 2.0
    
    pipeline = SignalPipeline(fs)
    lp_filter = LowPassFilter(cutoff_freq=10, order=2)
    
    processed_chunks = []
    processing_times = []
    
    print(f"Stream started: {duration}s signal, {chunk_size} samples/chunk")
    print("-" * 40)
    
    # Simulate Stream
    for i, chunk in enumerate(simulate_streaming_data(duration, chunk_size, fs)):
        start_time = time.time()
        
        # Process the chunk
        # Note: In a real real-time system, you would use 'overlap-add' 
        # to avoid edge artifacts. Here we simplify for demonstration.
        res = pipeline.run(chunk, filter_object=lp_filter)
        
        end_time = time.time()
        
        processed_chunks.append(res["output_signal"])
        processing_times.append((end_time - start_time) * 1000) # ms
        
        # Simulate live output
        sys.stdout.write(f"\rChunk {i+1}: Processed in {processing_times[-1]:.3f} ms")
        sys.stdout.flush()
        time.sleep(0.05) # Simulate sensor delay

    print("\n" + "-" * 40)
    print(f"Average Latency: {np.mean(processing_times):.3f} ms")
    
    # Stitch together for visualization
    full_output = np.concatenate(processed_chunks)
    
    plt.figure(figsize=(12, 4))
    plt.plot(full_output)
    plt.title("Reconstructed Stream Output")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.show()

if __name__ == "__main__":
    run_realtime_demo()