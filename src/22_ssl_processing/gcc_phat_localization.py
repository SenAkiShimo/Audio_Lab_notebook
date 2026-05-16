import numpy as np

def execute_gcc_phat_localization(signal_1: np.ndarray, signal_2: np.ndarray, mic_distance=0.02, fs=16000) -> float:
    """
    Computes Time Difference of Arrival (TDOA) and estimates the Incident Angle 
    using the Generalized Cross-Correlation with Phase Transform (GCC-PHAT) algorithm.
    
    signal_1, signal_2: Two-channel parallel mono audio signals containing spatial delay.
    mic_distance: Physical distance between microphones in meters (d).
    fs: Discrete sampling rate of the system in Hz.
    """
    v_sound = 340.0  # Speed of sound in m/s
    N = len(signal_1)
    
    # 1. Transform signals to the frequency domain using 1D FFT
    X1 = np.fft.fft(signal_1)
    X2 = np.fft.fft(signal_2)
    
    # 2. Compute cross-power spectrum
    cross_power_spectrum = X1 * np.conj(X2)
    
    # 3. Apply PHAT weighting (Whitening Filter) to normalize amplitudes
    # Add a small epsilon (1e-9) to prevent division by zero in quiet intervals
    phat_denominator = np.abs(cross_power_spectrum) + 1e-9
    phat_weighting_filter = cross_power_spectrum / phat_denominator
    
    # 4. Transform back to the time domain using Inverse FFT (IFFT)
    cc_complex = np.fft.ifft(phat_weighting_filter)
    cc_real = np.real(cc_complex)
    
    # 5. Shift the zero-lag component to the center of the spectrum
    cc_shifted = np.fft.fftshift(cc_real)
    
    # Construct discrete sample delay axis [-N/2, ..., +N/2-1]
    sample_delays = np.arange(-N // 2, N // 2)
    
    # 6. Locate the peak index corresponding to the dominant time delay
    peak_index = np.argmax(cc_shifted)
    estimated_sample_delay = sample_delays[peak_index]
    
    # 7. Convert discrete sample delay back to physical TDOA (seconds)
    estimated_tau = estimated_sample_delay / fs
    
    # 8. Calculate the spatial incident angle theta using inverse trigonometry
    # Formula: sin(theta) = tau * v / d -> theta = arcsin(tau * v / d)
    sin_theta = (estimated_tau * v_sound) / mic_distance
    
    # Clip values to [-1.0, 1.0] to prevent mathematical errors in arcsin due to noise
    sin_theta = np.clip(sin_theta, -1.0, 1.0)
    estimated_angle_deg = np.degrees(np.arcsin(sin_theta))
    
    print(f"[GCC-PHAT SUCCESS] Peak captured successfully.")
    print(f" -> Sample delay: {estimated_sample_delay} samples | Physical TDOA: {estimated_tau * 1e6:.2f} μs")
    print(f" -> Estimated Incident Angle: {estimated_angle_deg:.2f}°")
    
    return estimated_angle_deg

# Test driver configuration
if __name__ == "__main__":
    print("=" * 60)
    print("[RUN] Testing Python GCC-PHAT Source Localization Kernel...")
    
    # Simulate 1 second of audio stream at 16kHz sampling rate
    fs_sim = 16000
    t_axis = np.arange(0, 0.05, 1.0 / fs_sim)
    
    # Generate a pure tone source signal (800Hz)
    clean_source = np.sin(2 * np.pi * 800.0 * t_axis)
    
    # Simulate a 1-sample delay on channel 2
    channel_1_pcm = clean_source
    channel_2_pcm = np.roll(clean_source, shift=1)
    
    # Execute the GCC-PHAT localization solver
    detected_angle = execute_gcc_phat_localization(channel_1_pcm, channel_2_pcm, mic_distance=0.02, fs=fs_sim)
    print("=" * 60)
