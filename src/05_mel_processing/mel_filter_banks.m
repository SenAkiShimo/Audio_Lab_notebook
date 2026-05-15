function mel_fb = mel_filter_banks(num_mels, n_fft, fs)
    % Self-test execution block when run directly without input arguments
    if nargin == 0
        clc;
        disp('============================================================');
        disp('[RUN] Activating MATLAB pure matrix Mel-filter bank generator test driver...');
        
        num_mels_test = 80;
        n_fft_test = 1024;
        fs_test = 16000;
        
        mel_fb = mel_filter_banks(num_mels_test, n_fft_test, fs_test);
        [rows, cols] = size(mel_fb);
        
        disp('[SUCCESS] Psychoacoustic transfer matrix reconstructed.');
        disp([' -> Mel-filter bank matrix dimensions (Shape): ', num2str(rows), ' x ', num2str(cols)]);
        disp([' -> Target Mel bins: ', num2str(rows), ' | Linear FFT bins: ', num2str(cols)]);
        disp('============================================================');
        return;
    end

    % --- Core Algorithm Execution ---
    
    % 1. Define physical frequency boundaries
    f_min = 0;
    f_max = fs / 2;
    
    % 2. Hertz (Hz) to Mel scale logarithmic mapping
    mel_min = 2595 * log10(1 + f_min / 700);
    mel_max = 2595 * log10(1 + f_max / 700);
    
    % 3. Generate linearly spaced points in the Mel domain
    mel_points = linspace(mel_min, mel_max, num_mels + 2);
    
    % 4. Mel to Hertz (Hz) inverse mapping to get physical frequencies
    hz_points = 700 * (10.^(mel_points / 2595) - 1);
    
    % 5. Map physical frequencies to discrete FFT bin indices (Adjusted for MATLAB 1-based indexing)
    bin_points = floor((n_fft + 1) * hz_points / fs) + 1;
    
    % Initialize the filter bank transfer matrix
    num_fft_bins = floor(n_fft / 2) + 1;
    mel_fb = zeros(num_mels, num_fft_bins);
    
    % 6. Construct the triangular filter weights
    for m = 1:num_mels
        bin_left   = bin_points(m);
        bin_center = bin_points(m + 1);
        bin_right  = bin_points(m + 2);
        
        % Generate weights for the rising slope of the triangle
        for k = bin_left:(bin_center - 1)
            if k >= 1 && k <= num_fft_bins && (bin_center - bin_left) > 0
                mel_fb(m, k) = (k - bin_left) / (bin_center - bin_left);
            end
        end
        
        % Generate weights for the center and falling slope of the triangle
        for k = bin_center:bin_right
            if k >= 1 && k <= num_fft_bins && (bin_right - bin_center) > 0
                mel_fb(m, k) = (bin_right - k) / (bin_right - bin_center);
            end
        end
    end
end
