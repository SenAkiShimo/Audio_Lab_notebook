function [reverberant_signal, rir] = simulate_reverb(clean_signal, fs, room_dim, src_pos, mic_pos)
    % Check if there are no input parameters; if true, enter the self-test execution block.
    if nargin == 0
        clc;
        disp('============================================================');
        disp('[RUNNING SELF-TEST ENVIRONMENT]');
        fs_test = 16000;
        
        % Generate a distinct audio event (a short pulse burst) instead of pure white noise
        clean_signal_test = zeros(fs_test * 2, 1);
        clean_signal_test(100:500) = sin(2*pi*440*(0:400)/fs_test) * 0.5; 
        
        room_dim_test = [5, 4, 3]; 
        src_pos_test = [1, 1, 1.5]; 
        mic_pos_test = [3, 2, 1.5]; 
        
        % Invoke self for RIR calculation and convolution
        [reverberant_signal, rir] = simulate_reverb(clean_signal_test, fs_test, room_dim_test, src_pos_test, mic_pos_test);
        
        % Visual anchors for verification and troubleshooting inside the test block
        disp(['[DEBUG] RIR Length (Taps): ', num2str(length(rir))]);
        disp(['[DEBUG] RIR Absolute Maximum Value: ', num2str(max(abs(rir)))]);
        disp(['[DEBUG] Reverberant Signal Absolute Maximum: ', num2str(max(abs(reverberant_signal)))]);
        
        % Diagnostic Step 1: Export to a WAV file so you can listen to it
        audiowrite('test_output.wav', reverberant_signal, fs_test);
        disp('[SUCCESS] Audio file exported successfully as "test_output.wav"');
        
        % Diagnostic Step 2: Open a physical plot window to prove the data exists
        figure('Name', 'Acoustic Simulation Diagnostics');
        subplot(2,1,1);
        plot(rir, 'LineWidth', 1.5);
        title('Room Impulse Response (RIR) Filter Coefficients');
        xlabel('Samples'); ylabel('Amplitude'); grid on;
        
        subplot(2,1,2);
        plot(reverberant_signal, 'Color', [0.85 0.33 0.1]);
        title('Generated Reverberant Audio Signal');
        xlabel('Samples'); ylabel('Normalized Amplitude'); grid on;
        
        disp('============================================================');
        return;
    end

    % Acoustic Constants
    c = 340; 
    rt60 = 0.4;
    num_samples_rir = round(rt60 * fs);
    rir = zeros(num_samples_rir, 1);
    
    % 1. Direct Path Calculation
    dist_direct = norm(src_pos - mic_pos);
    time_direct = dist_direct / c;
    sample_direct = round(time_direct * fs) + 1;
    
    % Safe Array Expansion
    if sample_direct > length(rir)
        rir(sample_direct) = 0; 
    end
    rir(sample_direct) = 1.0 / max(dist_direct, 1.0);
    
    % 2. First-Order Reflection Calculation (Image Source Method)
    reflection_coeff = 0.65;
    walls = [0, room_dim(1); 0, room_dim(2); 0, room_dim(3)];
    
    for w = 1:3
        for side = 1:2
            virtual_src = src_pos;
            virtual_src(w) = 2 * walls(w, side) - src_pos(w);
            
            dist_reflect = norm(virtual_src - mic_pos);
            time_reflect = dist_reflect / c;
            sample_reflect = round(time_reflect * fs) + 1;
            
            if sample_reflect <= length(rir)
                rir(sample_reflect) = rir(sample_reflect) + (reflection_coeff / max(dist_reflect, 1.0));
            end
        end
    end
    
    % 3. Convolutional Reverberation Generation
    reverberant_signal = conv(clean_signal, rir);
    
    % 4. Safe Signal Normalization
    max_val = max(abs(reverberant_signal));
    if max_val > 0
        reverberant_signal = reverberant_signal / max_val;
    else
        warning('CRITICAL: The max value of the convoluted signal is 0. Please verify the validity of clean_signal.');
    end
end
