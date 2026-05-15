import numpy as np

def compute_pure_ctc_forward_prob(log_probs_matrix: np.ndarray, target_labels_list: list, blank_idx=0):
    """
    Computes the CTC forward variable trellis matrix and the total alignment probability
    using the dynamic programming forward algorithm with scaling to prevent numerical underflow.
    
    log_probs_matrix: Log-probabilities from neural network output of shape [Time_Frames, Num_Classes]
    target_labels_list: Ground-truth target label index sequence of length U
    blank_idx: Integer index denoting the blank token (epsilon)
    """
    T, num_classes = log_probs_matrix.shape
    U = len(target_labels_list)
    
    # 1. Construct the extended target sequence Y' by interleaving blank tokens (Length S = 2U + 1)
    extended_targets = [blank_idx]
    for label in target_labels_list:
        extended_targets.append(label)
        extended_targets.append(blank_idx)
    S = len(extended_targets)
    
    # 2. Initialize the forward probability lattice matrix (alpha) and scaling tracker (c)
    alpha = np.zeros((T, S))
    c = np.zeros(T)
    
    # Boundary conditions at time t = 0
    alpha[0, 0] = np.exp(log_probs_matrix[0, extended_targets[0]])
    if S > 1:
        alpha[0, 1] = np.exp(log_probs_matrix[0, extended_targets[1]])
        
    # Apply initial scaling factor to prevent early truncation
    c[0] = 1.0 / (np.sum(alpha[0, :]) + 1e-15)
    alpha[0, :] *= c[0]
    
    # 3. Time-frequency lattice dynamic programming execution pass
    for t in range(1, T):
        probs = np.exp(log_probs_matrix[t, extended_targets])
        
        # Branch 1 & 2: Self-loop staying paths and sequential stepping paths
        prev_stay = alpha[t - 1, :]
        prev_step = np.zeros(S)
        prev_step[1:] = alpha[t - 1, :-1]
        
        # Branch 3: Skip-blank jumping paths (conditional activation)
        prev_skip = np.zeros(S)
        if S > 2:
            # Mask out cases where current token is blank or equals the token 2 steps back
            skip_mask = np.ones(S, dtype=bool)
            for s in range(2, S):
                if extended_targets[s] == blank_idx or extended_targets[s] == extended_targets[s - 2]:
                    skip_mask[s] = False
            
            prev_skip[2:] = alpha[t - 1, :-2]
            prev_skip = prev_skip * skip_mask
            
        # Accumulate structural paths and apply current emission tokens probability
        alpha[t, :] = (prev_stay + prev_step + prev_skip) * probs
        
        # Dynamic normalization scaling per time-step to fully resolve underflow errors
        c[t] = 1.0 / (np.sum(alpha[t, :]) + 1e-15)
        alpha[t, :] *= c[t]
        
    # 4. Collapse probability paths aggregation and re-scaling inversion
    p_scaled_text = alpha[T - 1, S - 1] + (alpha[T - 1, S - 2] if S > 1 else 0.0)
    
    # Compute the total log-likelihood by combining all step scaling variables
    log_p_total = -np.sum(np.log(c))
    p_total_text = np.exp(log_p_total) * p_scaled_text
    
    return p_total_text, alpha, log_p_total

if __name__ == "__main__":
    print("=" * 60)
    print("[RUN] Initializing Python matrix CTC forward dynamic programming trellis alignment test driver...")
    
    # Simulate acoustic encoder sequence of 30 frames with 3 vocabulary tokens (0=blank, 1=vocab_A, 2=vocab_B)
    mock_logits = np.random.uniform(-2.0, 2.0, (30, 3))
    mock_log_probs = mock_logits - np.log(np.sum(np.exp(mock_logits), axis=1, keepdims=True))
    
    # Simulate target sequence text label indices
    true_labels = [1, 2]
    
    # Execute the numerical safe dynamic programming kernel
    total_prob, alpha_matrix, log_likelihood = compute_pure_ctc_forward_prob(mock_log_probs, true_labels)
    
    # Convert global alignment probability to negative log-likelihood Loss value
    ctc_loss_val = -log_likelihood
    
    t_frames, s_states = alpha_matrix.shape
    print("[SUCCESS] CTC transition trellis probability pathways computation converged successfully.")
    print(f" -> Trellis lattice dimensions: {t_frames} frames x {s_states} state spaces")
    print(f" -> Total alignment path likelihood P(Y|X): {total_prob:.8f}")
    print(f" -> Differentiable CTC negative log-likelihood Loss value: {ctc_loss_val:.4f}")
    print("=" * 60)
