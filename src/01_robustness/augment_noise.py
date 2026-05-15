import numpy as np

def inject_additive_gaussian_noise(clean_waveform: np.ndarray, target_snr_db: float) -> np.ndarray:
    """
    Precise SNR-Targeted Gaussian Noise Injection via Pure NumPy
    """
    signal_power = np.mean(clean_waveform ** 2)
    target_snr_linear = 10 ** (target_snr_db / 10.0)
    required_noise_power = signal_power / target_snr_linear
    
    noise_std_dev = np.sqrt(required_noise_power)
    gaussian_noise = np.random.normal(0.0, noise_std_dev, len(clean_waveform))
    
    noisy_waveform = clean_waveform + gaussian_noise
    noisy_waveform = np.clip(noisy_waveform, -1.0, 1.0)
    
    return noisy_waveform

if __name__ == "__main__":
    print("=" * 60)
    print("[RUN]")
    
    t = np.linspace(0, 1, 16000)
    mock_clean_audio = np.sin(2 * np.pi * 440.0 * t) * 0.5  # 440Hz
    
    # -5dB SNR 
    target_snr = -5.0
    output_audio = inject_additive_gaussian_noise(mock_clean_audio, target_snr)
    
    print(f"[SUCCESS] Enter Pure Points: {len(mock_clean_audio)} | Target Signal-to-Noise Ratio: {target_snr} dB")
    print("=" * 60)
