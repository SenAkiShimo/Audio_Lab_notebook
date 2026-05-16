function [aliasing_matrix, critical_freq] = spatial_aliasing_sim(mic_distance, max_freq, fs)
    % Self-test driver configuration
    if nargin == 0
        clc;
        disp('============================================================');
        disp('[RUN] Testing Spatial Aliasing Frequency Response Simulator...');
        
        % Simulation parameters: 5cm mic spacing, 8kHz max frequency, 16kHz sampling rate
        mic_d_test = 0.05;
        max_f_test = 8000;
        fs_test = 16000;
        
        [aliasing_matrix, critical_freq] = spatial_aliasing_sim(mic_d_test, max_f_test, fs_test);
        
        disp('[SUCCESS] Spatial aliasing simulation completed.');
        disp([' -> Critical spatial Nyquist frequency: ', num2str(critical_freq), ' Hz']);
        disp('============================================================');
        return;
    end

    % --- Core Algorithm Execution ---
    v_sound = 340; % Speed of sound in m/s
    
    % Calculate the spatial Nyquist cutoff frequency based on the microphone distance
    % Criteria: d <= v / (2 * Fmax) -> F_critical = v / (2 * d)
    critical_freq = v_sound / (2 * mic_distance);
    
    % Define discrete frequency and spatial scanning grids
    freq_axis = 100:100:max_freq;
    scan_angles = -90:5:90; 
    aliasing_matrix = zeros(length(scan_angles), length(freq_axis));

    % Simulate spatial phase differences and analyze aliasing conditions
    for f_idx = 1:length(freq_axis)
        f = freq_axis(f_idx);
        wavelength = v_sound / f;
        
        for a_idx = 1:length(scan_angles)
            theta = deg2rad(scan_angles(a_idx));
            
            % Compute the spatial phase difference between adjacent microphones
            % Phase_Diff = 2 * pi * d * sin(theta) / lambda
            phase_difference = (2 * pi * mic_distance * sin(theta)) / wavelength;
            
            % Check spatial aliasing condition: phase difference exceeds pi (phase ambiguity)
            if abs(phase_difference) > pi
                % Apply a negative penalty score proportional to the degree of aliasing
                aliasing_matrix(a_idx, f_idx) = -20.0 * (abs(phase_difference) / pi);
            else
                % No spatial aliasing observed within this configuration
                aliasing_matrix(a_idx, f_idx) = 0.0;
            end
        end
    end
end
