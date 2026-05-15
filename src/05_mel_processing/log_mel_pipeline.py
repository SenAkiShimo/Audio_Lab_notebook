import numpy as np

def execute_log_mel_compression(stft_power_spectrogram: np.ndarray, mel_filter_bank_matrix: np.ndarray) -> np.ndarray:
    """
    Applies spatial dimensionality reduction on an STFT power spectrogram 
    using a Mel-filter bank matrix, followed by log-amplitude compression.
    
    stft_power_spectrogram: Matrix of shape [FFT_Bins, Time_Frames]
    mel_filter_bank_matrix: Matrix of shape [Num_Mels, FFT_Bins]
    """
    # 1. Dimensionality reduction via matrix multiplication: [Num_Mels, FFT_Bins] * [FFT_Bins, Time_Frames] -> [Num_Mels, Time_Frames]
    mel_spectrogram = np.dot(mel_filter_bank_matrix, stft_power_spectrogram)
    
    # 2. Logarithmic amplitude compression with an epsilon floor to prevent negative infinity overflow
    log_mel = 10.0 * np.log10(mel_spectrogram + 1e-10)
    
    # 3. Dynamic range clipping: Restrict the range to 80 dB below the global maximum energy
    max_db = np.max(log_mel)
    log_mel_clipped = np.maximum(log_mel, max_db - 80.0)
    
    return log_mel_clipped

if __name__ == "__main__":
    print("=" * 60)
    print("[RUN] Initializing Python matrix Mel-reduction and log-compression pipeline test driver...")
    
    # Simulate an STFT power spectrogram matrix with 513 bins (1024-point FFT) and 400 time frames
    mock_power_spec = np.random.uniform(0, 1, (513, 400)) ** 2
    
    # Simulate a Mel-filter bank matrix with 80 Mel bands and 513 linear FFT bins
    mock_mel_basis = np.random.uniform(0, 0.5, (80, 513))
    
    # Corrected parameter passing order to match matrix multiplication alignment
    final_log_mel = execute_log_mel_compression(mock_power_spec, mock_mel_basis)
    mels, frames = final_log_mel.shape
    
    print("[SUCCESS] Log-Mel dense feature extraction completed.")
    print(f" -> Output feature matrix shape: {mels} x {frames}")
    print(f" -> Dynamic range validation: Max = {np.max(final_log_mel):.2f} dB | Min = {np.min(final_log_mel):.2f} dB")
    print("=" * 60)
