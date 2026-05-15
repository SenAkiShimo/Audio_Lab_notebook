import numpy as np

def compute_pure_dft_matrix(time_frame: np.ndarray) -> np.ndarray:
    """
    Computes the 1D Discrete Fourier Transform (DFT) using Euler's formula
    via matrix multiplication, returning the single-sided complex spectrum.
    """
    N = len(time_frame)
    
    # Construct the discrete coordinate grid for basis interactions
    n = np.arange(N)
    k = n.reshape((N, 1))
    
    # Apply Euler's relation for the complex rotation factor: exp(-j * 2 * pi * k * n / N)
    exponent_matrix = -2j * np.pi * k * n / N
    dft_kernel_matrix = np.exp(exponent_matrix)
    
    # Project time-domain signal into the complex frequency domain
    complex_spectrum = np.dot(dft_kernel_matrix, time_frame)
    
    # Retain the single-sided spectrum based on Nyquist symmetry
    return complex_spectrum[:N // 2 + 1]

def generate_log_spectrogram(framed_signals_list: list) -> np.ndarray:
    """
    Processes a list of time-domain frames to compute and stack their 
    corresponding log-magnitude spectral representations.
    """
    spectrogram_list = []
    
    for frame in framed_signals_list:
        # 1. Compute complex spectrum using the custom DFT matrix kernel
        complex_spec = compute_pure_dft_matrix(frame)
        
        # 2. Extract magnitude spectrum: Magnitude = sqrt(real^2 + imag^2)
        magnitude_spec = np.abs(complex_spec)
        
        # 3. Apply logarithmic compression and define a dynamic range floor at -80 dB
        log_magnitude = 20 * np.log10(magnitude_spec + 1e-9)
        log_magnitude = np.maximum(log_magnitude, -80.0)
        
        spectrogram_list.append(log_magnitude)
        
    # Concatenate frames horizontally into a [Frequency_Bins, Time_Frames] matrix
    return np.column_stack(spectrogram_list)

if __name__ == "__main__":
    print("=" * 60)
    print("[RUN] Initializing Python matrix DFT kernel and log-spectrogram conversion test driver...")
    
    # Simulate 5 short-time frames (each 256 samples long) pre-processed with a window function
    mock_frames = [np.sin(np.linspace(0, 5, 256)) for _ in range(5)]
    
    # Execute time-frequency representation transformation
    spectrogram_matrix = generate_log_spectrogram(mock_frames)
    freq_bins, time_frames = spectrogram_matrix.shape
    
    print("[SUCCESS] Two-dimensional time-frequency log-spectrogram matrix generated.")
    print(f" -> Matrix dimensions (Shape): {freq_bins} x {time_frames}")
    print(f" -> Single-sided frequency channels (Bins): {freq_bins}")
    print("=" * 60)
