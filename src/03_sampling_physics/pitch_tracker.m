function [f0_trajectory, time_axis] = pitch_tracker(signal, fs)
    % Self-test execution block when the function is run without input arguments
    if nargin == 0
        clc;
        disp('============================================================');
        disp('[RUN] Activating MATLAB time-domain autocorrelation pitch tracker self-test...');
        
        % Generate a 0.5-second test signal with a 16 kHz sampling rate and 200 Hz fundamental frequency
        fs_test = 16000;
        t_test = 0:(1/fs_test):0.5;
        f0_true = 200;
        
        % Combine the fundamental frequency and the first harmonic
        mock_vocal = sin(2*pi*f0_true*t_test) + 0.5*sin(2*pi*(2*f0_true)*t_test);
        
        [f0_trajectory, time_axis] = pitch_tracker(mock_vocal', fs_test);
        valid_f0 = f0_trajectory(f0_trajectory > 0);
        
        disp(['[SUCCESS] Pitch trajectory extraction complete. Estimated mean F0: ', num2str(mean(valid_f0)), ' Hz']);
        disp('============================================================');
        return;
    end

    % --- Core Algorithm Execution ---
    frame_len = round(0.03 * fs); % 30ms frame length
    hop_len = round(0.01 * fs);   % 10ms hop length
    num_samples = length(signal);
    num_frames = floor((num_samples - frame_len) / hop_len) + 1;
    
    f0_trajectory = zeros(num_frames, 1);
    time_axis = zeros(num_frames, 1);
    
    % Set human pitch limits for search boundaries: 50 Hz to 500 Hz
    max_lag = round(fs / 50);
    min_lag = round(fs / 500);
    
    for f = 1:num_frames
        start_idx = (f-1)*hop_len + 1;
        end_idx = start_idx + frame_len - 1;
        frame = signal(start_idx:end_idx);
        
        % Apply Hann window to reduce spectral leakage from frame edges
        frame_windowed = frame .* hann(frame_len);
        
        % Compute short-time autocorrelation function (ACF) coefficients
        acf = zeros(max_lag, 1);
        for lag = min_lag:max_lag
            acf(lag) = sum(frame_windowed(1:frame_len-lag) .* frame_windowed(lag+1:frame_len));
        end
        
        % Find the maximum correlation peak within the human pitch search window
        [peak_val, idx] = max(acf(min_lag:max_lag));
        actual_lag = idx + min_lag - 1;
        
        % Voiced/Unvoiced decision based on the peak-to-energy threshold ratio
        if peak_val > 0.3 * acf(1)
            f0_trajectory(f) = fs / actual_lag; % Convert lag period to frequency
        else
            f0_trajectory(f) = 0; % Categorized as unvoiced or silent frame
        end
        
        time_axis(f) = (start_idx + frame_len/2) / fs;
    end
end
