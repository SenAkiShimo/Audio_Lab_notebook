import numpy as np

def compute_pure_mas_matrix(log_likelihood_matrix: np.ndarray):
    """
    Computes the optimal monotonic alignment path between text phonemes and audio frames
    using the VITS Monotonic Alignment Search (MAS) dynamic programming framework.
    
    log_likelihood_matrix: Tensor of shape [U, T]
    U: Number of text phonemes
    T: Number of audio frames
    """
    U, T = log_likelihood_matrix.shape
    
    # 1. Initialize the dynamic programming cumulative maximum log-likelihood matrix (Q)
    Q = np.full((U, T), -np.inf)
    
    # Anchor the start boundary condition at origin (t=0, u=0)
    Q[0, 0] = log_likelihood_matrix[0, 0]
    
    # 2. Forward dynamic programming recursion across the 2D grid matrix
    for t in range(1, T):
        # Scan across all phonemes with a safe boundary buffer to guarantee alignment convergence
        for u in range(max(0, U - (T - t)), U):
            current_ll = log_likelihood_matrix[u, t]
            
            # Enforce MAS constraints: state can only transition from current phoneme (stay) or previous phoneme (step)
            prev_stay = Q[u, t - 1]
            prev_step = Q[u - 1, t - 1] if u > 0 else -np.inf
            
            # Select the maximum likelihood pathway and accumulate current emission log-probability
            max_prev = max(prev_stay, prev_step)
            if max_prev != -np.inf:
                Q[u, t] = max_prev + current_ll
                
    # 3. Backward path verification and hard-alignment matrix reconstruction
    alignment_matrix = np.zeros((U, T))
    
    # Terminal anchoring constraint: force path convergence at the last phoneme (U-1) at the final frame (T-1)
    current_u = U - 1
    alignment_matrix[current_u, T - 1] = 1.0
    
    # Trace the lattice matrix backward from right to left along the timeline
    for t in range(T - 2, -1, -1):
        stay_val = Q[current_u, t]
        step_val = Q[current_u - 1, t] if current_u > 0 else -np.inf
        
        # Decide whether the path shifted phonemes or stayed in place at time step t
        if step_val > stay_val:
            current_u -= 1
            
        alignment_matrix[current_u, t] = 1.0
        
    return alignment_matrix, Q

if __name__ == "__main__":
    print("=" * 60)
    print("[RUN] Initializing Python matrix VITS-MAS dynamic programming alignment test driver...")
    
    # Simulate a system containing U = 6 text phonemes and T = 20 audio frames
    U_test, T_test = 6, 20
    mock_ll_matrix = np.full((U_test, T_test), -10.0)
    
    # Forge an artificial high-energy diagonal ridge representing a correct alignment tendency
    for t in range(T_test):
        target_u = min(int(t * (U_test / T_test)), U_test - 1)
        mock_ll_matrix[target_u, t] = 0.0  # 0.0 corresponds to a probability of 100% in log space
        if target_u > 0:
            mock_ll_matrix[target_u - 1, t] = -1.5
        if target_u < U_test - 1:
            mock_ll_matrix[target_u + 1, t] = -1.5
            
    # Execute the MAS alignment search
    alignment, score_Q = compute_pure_mas_matrix(mock_ll_matrix)
    
    print("[SUCCESS] VITS hard-alignment tracking path computation completed.")
    print(f" -> Accumulation matrix dimensions (Shape): {score_Q.shape}")
    print(f" -> Extracted hard alignment trajectory mapping (First 10 frames):")
    
    matched_u_indices = np.argmax(alignment, axis=0)
    print(f" -> Time frames t [0~9] mapped phoneme indices: {matched_u_indices[:10].tolist()}")
    print("=" * 60)