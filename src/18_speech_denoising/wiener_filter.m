function enhanced_stft = wiener_filter(noisy_stft)
    % Self-test execution driver block
    if nargin == 0
        clc;
        disp('============================================================');
        disp('[RUN] Activating MATLAB time-frequency domain adaptive Wiener filter denoising test...');
        
        % Simulate a complex noisy STFT matrix: 257 frequency bins (512-point FFT) x 300 time frames
        freq_bins_test = 257;
        time_frames_test = 300;
        mock_real = randn(freq_bins_test, time_frames_test);
        mock_imag = randn(freq_bins_test, time_frames_test);
        mock_stft = mock_real + 1i * mock_imag;
        
        % Execute the Wiener filter
        enhanced_stft = wiener_filter(mock_stft);
        [rows, cols] = size(enhanced_stft);
        
        disp('[SUCCESS] Wiener gain envelope calculation complete.');
        disp([' -> Enhanced complex matrix shape: ', num2str(rows), ' x ', num2str(cols)]);
        disp('============================================================');
        return;
    end

    % ─── Core Algorithm ───
    [freq_bins, time_frames] = size(noisy_stft);

    % 1. Calculate magnitude and power spectrum matrices
    noisy_magnitude = abs(noisy_stft);
    noisy_power = noisy_magnitude .^ 2;

    % 2. Noise power estimation
    % Assume the first 5 frames are non-speech noise-only periods to calculate a static profile
    noise_frames_estimate = noisy_power(:, 1:5);
    estimated_noise_power = mean(noise_frames_estimate, 2);

    % 3. Broadcast the noise profile matrix across the time axis
    noise_power_broadcast = repmat(estimated_noise_power, 1, time_frames);

    % 4. Estimate clean speech power using adaptive spectral subtraction approximation
    % Apply a small bias ceiling to prevent negative power values
    estimated_clean_power = max(noisy_power - noise_power_broadcast, 1e-9);

    % 5. Derive the adaptive Wiener filter gain matrix G(f, t)
    % Formula: G = P_clean / (P_clean + P_noise)
    wiener_gain_matrix = estimated_clean_power ./ (estimated_clean_power + noise_power_broadcast);

    % 6. Apply gain weights to the original complex matrix (maintaining micro-phase profiles)
    enhanced_stft = wiener_gain_matrix .* noisy_stft;
end
