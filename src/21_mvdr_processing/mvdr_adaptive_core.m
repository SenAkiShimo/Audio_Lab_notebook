function [w_mvdr, filtered_fft] = mvdr_adaptive_core(multichannel_fft_data, target_angle_deg, mic_distance, fs, freq_hz)
    % Self-test driver configuration
    if nargin == 0
        clc;
        disp('============================================================');
        disp('[RUN] Testing Complex-Matrix Inversion MVDR Adaptive Beamformer...');
        
        % Simulation parameters: 4 mics, 300 snapshots, 2cm spacing, 16kHz sampling rate
        num_mics_test = 4;
        num_snapshots_test = 300; 
        mic_d_test = 0.02;
        fs_test = 16000;
        freq_test = 1000; % Evaluate at 1000Hz frequency bin
        
        % Generate mock multichannel complex FFT data containing noise and interference
        mock_real = randn(num_mics_test, num_snapshots_test);
        mock_imag = randn(num_mics_test, num_snapshots_test);
        mock_multichannel_fft = mock_real + 1i * mock_imag;
        
        % Execute MVDR beamforming assuming the target source is at 0 degrees
        [w_mvdr, filtered_fft] = mvdr_adaptive_core(mock_multichannel_fft, 0.0, mic_d_test, fs_test, freq_test);
        
        disp('[SUCCESS] MVDR adaptive complex weights calculated successfully.');
        disp([' -> Optimal weight vector dimensions: ', num2str(length(w_mvdr)), ' x 1']);
        disp('============================================================');
        return;
    end

    % --- Core Algorithm Execution ---
    [num_mics, num_snapshots] = size(multichannel_fft_data);
    theta_rad = deg2rad(target_angle_deg);
    omega = 2 * pi * freq_hz;
    v_sound = 340; 

    % 1. Construct the steering vector (d) for the target source direction
    d = zeros(num_mics, 1);
    for m = 1:num_mics
        tau = (m-1) * mic_distance * sin(theta_rad) / v_sound;
        d(m) = exp(-1i * omega * tau);
    end

    % 2. Compute the spatial covariance matrix R_xx
    % Formula: R = (X * X^H) / Snapshots
    R = (multichannel_fft_data * multichannel_fft_data') / num_snapshots;

    % 3. Apply diagonal loading to stabilize matrix inversion
    % This prevents ill-conditioned or singular matrix states in low-noise or quiet intervals
    diagonal_loading_factor = 1e-5;
    R = R + diagonal_loading_factor * eye(num_mics);

    % 4. Perform matrix inversion (R^-1)
    R_inv = inv(R);

    % 5. Compute MVDR optimal weight vector using Lagrange multipliers
    % Formula: w_mvdr = (R^-1 * d) / (d^H * R^-1 * d)
    numerator = R_inv * d;
    denominator = d' * R_inv * d; 
    w_mvdr = numerator / denominator;

    % 6. Spatial filtering: Apply optimal weights to filter the multi-channel sequence
    % Formula: Y(f, t) = w^H * X(f, t)
    filtered_fft = w_mvdr' * multichannel_fft_data;
end
