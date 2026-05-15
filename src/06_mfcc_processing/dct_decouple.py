import numpy as np

def compute_vectorized_dct_ii(log_mel_spectrogram: np.ndarray, n_mfcc=13) -> np.ndarray:
    """
    Computes the Discrete Cosine Transform (DCT-II) using vectorized matrix 
    multiplication to perform global de-correlation across all time frames.
    
    log_mel_spectrogram: Matrix of shape [Num_Mels, Time_Frames]
    """
    num_mels, time_frames = log_mel_spectrogram.shape
    
    # Construct orthogonal basis matrix grid of shape [n_mfcc, num_mels]
    k = np.arange(n_mfcc).reshape((n_mfcc, 1))
    m = np.arange(num_mels)
    ortho_basis_matrix = np.cos(np.pi * k * (m + 0.5) / num_mels)
    
    # Compute orthogonal normalization scale factors
    scale_factors = np.sqrt(2.0 / num_mels) * np.ones((n_mfcc, 1))
    scale_factors[0, 0] = np.sqrt(1.0 / num_mels)
    
    # Map all frames into the cepstral domain using a single matrix multiplication pass
    raw_weights = np.dot(ortho_basis_matrix, log_mel_spectrogram)
    dct_coefficients = raw_weights * scale_factors
    
    return dct_coefficients

def compute_time_delta(static_matrix: np.ndarray) -> np.ndarray:
    """
    Computes the first-order dynamic time delta coefficients.
    Includes boundary protection to prevent edge value truncation.
    """
    dims, frames = static_matrix.shape
    delta_matrix = np.zeros_like(static_matrix)
    
    if frames <= 1:
        return delta_matrix
        
    # Calculate velocity weights for internal continuous frames
    for t in range(1, frames - 1):
        delta_matrix[:, t] = (static_matrix[:, t + 1] - static_matrix[:, t - 1]) / 2.0
        
    # Apply edge padding to handle boundary frames securely
    delta_matrix[:, 0] = static_matrix[:, 1] - static_matrix[:, 0]
    delta_matrix[:, frames - 1] = static_matrix[:, frames - 1] - static_matrix[:, frames - 2]
    
    return delta_matrix

def generate_39d_mfcc_pipeline(log_mel_spectrogram: np.ndarray) -> np.ndarray:
    """
    Cascades 13-dimensional static MFCCs, 13-dimensional delta, and 
    13-dimensional delta-delta features into a unified 39-dimensional matrix.
    """
    # 1. High-speed vectorized space mapping and dimensionality reduction
    mfcc_static = compute_vectorized_dct_ii(log_mel_spectrogram, n_mfcc=13)
    
    # 2. Extract first-order temporal velocity features
    mfcc_delta = compute_time_delta(mfcc_static)
    
    # 3. Extract second-order temporal acceleration features
    mfcc_delta_delta = compute_time_delta(mfcc_delta)
    
    # 4. Vertical stack combination to form the standard 39D feature container
    mfcc_39d = np.vstack([mfcc_static, mfcc_delta, mfcc_delta_delta])
    return mfcc_39d

if __name__ == "__main__":
    print("=" * 60)
    print("[RUN] Initializing Python matrix DCT-II and 39D dynamic delta pipeline test driver...")
    
    # Simulate a log-mel spectrogram matrix with 40 channels and 300 frames
    mock_log_mel = np.random.uniform(-40.0, 0.0, (40, 300))
    
    # Execute feature extraction pipeline
    final_39d_matrix = generate_39d_mfcc_pipeline(mock_log_mel)
    channels, frames = final_39d_matrix.shape
    
    print("[SUCCESS] 39-dimensional dynamic cepstral feature matrix extraction completed.")
    print(f" -> Output matrix dimensions (Shape): {channels} x {frames}")
    print(f" -> Feature composition: 13D Static + 13D Delta + 13D Delta-Delta = {channels}")
    print("=" * 60)
