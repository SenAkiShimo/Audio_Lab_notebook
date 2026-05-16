function modgd_spectrum = phase_modgd_analyzer(time_signal, fs)
    % Self-test driver configuration
    if nargin == 0
        clc;
        disp('============================================================');
        disp('[RUN] Testing Modified Group Delay (MODGD) Feature Extractor...');
        
        % Simulate 0.1 seconds of mono audio input at 16kHz sampling rate
        fs_test = 16000;
        t_test = 0:(1/fs_test):0.1;
        mock_wave = sin(2*pi*300*t_test) + 0.2*randn(size(t_test));
        
        % Ensure the input signal is processed as a column vector
        modgd_spectrum = phase_modgd_analyzer(mock_wave(:), fs_test);
        [rows, cols] = size(modgd_spectrum);
        
        disp('[SUCCESS] MODGD high-order phase feature matrix calculation completed.');
        disp([' -> Feature matrix dimensions: ', num2str(rows), ' frequency bins x ', num2str(cols), ' frames']);
        disp('============================================================');
        return;
    end

    % --- Core Algorithm Execution ---
    % Ensure input signal is a column vector
    time_signal = time_signal(:);
    
    N = length(time_signal);
    n_fft = 512;
    frame_len = round(0.025 * fs);      % 25ms frame length
    hop_len = round(0.010 * fs);       % 10ms frame shift
    num_frames = floor((N - frame_len) / hop_len) + 1;
    num_fft_bins = floor(n_fft / 2) + 1;
    
    modgd_spectrum = zeros(num_fft_bins, num_frames);
    
    % Scaling parameter to smooth out zero-point spikes in the spectrum
    gamma = 0.5; 
    time_ramp = (0:frame_len-1)';

    % Process short-time frames
    for f = 1:num_frames
        start_idx = (f-1)*hop_len + 1;
        end_idx = start_idx + frame_len - 1;
        frame = time_signal(start_idx:end_idx);
        
        % 1. Compute standard FFT of the windowed frame
        X_complex = fft(frame, n_fft);
        X_real = real(X_complex(1:num_fft_bins));
        X_imag = imag(X_complex(1:num_fft_bins));
        
        % 2. Compute FFT of the time-weighted frame to derive phase derivatives
        time_sloped_frame = frame .* time_ramp;
        Y_complex = fft(time_sloped_frame, n_fft);
        Y_real = real(Y_complex(1:num_fft_bins));
        Y_imag = imag(Y_complex(1:num_fft_bins));
        
        % 3. Calculate and smooth the power spectrum to prevent division by zero
        power_spectrum = abs(X_complex(1:num_fft_bins)) .^ 2;
        smoothed_power = max(power_spectrum, 1e-8); 
        
        % 4. Reconstruct the Modified Group Delay spectrum
        % Formula: tau = (X_R * Y_R + X_I * Y_I) / (smoothed_power ^ (2 * gamma))
        numerator = X_real .* Y_real + X_imag .* Y_imag;
        denominator = smoothed_power .^ (2 * gamma);
        
        modgd_spectrum(:, f) = numerator ./ denominator;
    end
end
