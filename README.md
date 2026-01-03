# A Fourier-Domain Framework for Generalized Signal Filtering and Denoising

**Author:** Daksh Saini  

---

## Project Overview

Noise contamination is a universal challenge in signal processing, affecting data integrity in fields ranging from biomedical engineering to astrophysics. Existing tools are often "domain-specific," relying on fixed noise models that fail when applied to novel datasets.

This project presents a **General-Purpose Fourier-Domain Framework** designed to democratize signal analysis. By leveraging the **Fast Fourier Transform (FFT)** and a modular filter design engine, this system provides a mathematically rigorous and computationally efficient tool for analyzing and denoising diverse one-dimensional datasets.

### Key Capabilities
* **Domain Agnostic:** Proven effectiveness on ECG (Biomedical), P-waves (Seismic), and Audio signals.
* **Spectral Precision:** Manipulates signals in the frequency domain using $O(N \log N)$ FFT algorithms.
* **Quantitative Validation:** Built-in calculation of **SNR** (Signal-to-Noise Ratio), **MSE** (Mean Squared Error), and **PSD** (Power Spectral Density).

---

##  System Architecture

The framework is modularized into four distinct processing units:

1.  **Preprocessing Module:** Handles normalization, detrending, and windowing (Hann/Hamming) to mitigate spectral leakage.
2.  **Spectral Transformation Core:** Executes FFT/IFFT operations and computes magnitude/phase spectra.
3.  **Filter Design Unit:** A configurable engine for creating digital filters:
    * *Low-Pass / High-Pass* (Butterworth & Ideal)
    * *Band-Pass / Band-Stop*
    * *Notch Filters* (Specific frequency rejection, e.g., 60Hz mains hum)
    * *Adaptive Spectral Subtraction*
4.  **Reconstruction & Evaluation:** Rebuilds time-domain signals and computes error metrics.

---

##  Installation & Setup

### Prerequisites
* **OS:** Windows, Linux, or macOS
* **Python:** Version 3.10 or higher
* **RAM:** Minimum 8 GB recommended

### Quick Start
1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/arachknight-daksh/Fourier_Tool.git](https://github.com/arachknight-daksh/Fourier_Tool.git)
    cd Fourier_Tool
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run Validation Script:**
    Verify the core logic with the automated test suite:
    ```bash
    python run_validation.py
    ```

---

##  Usage Guide

### Running the Jupyter Notebooks
The project includes a comprehensive 6-part notebook series demonstrating the theory and application:

```bash
jupyter notebook
