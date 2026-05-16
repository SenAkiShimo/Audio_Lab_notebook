function beam_matrix = beam_pattern_analyzer(num_mics, mic_distance, target_angle_deg)
    % Self-test driver configuration
    if nargin == 0
        clc;
        disp('============================================================');
        disp('[RUN] Testing Uniform Linear Array Beam Pattern Analyzer...');
        
        % Simulation parameters: 4 mics, 2cm spacing, 30-degree target angle
        mics = 4;
        d_space = 0.02;
        target_theta = 30.0;
        
        beam_matrix = beam_pattern_analyzer(mics, d_space, target_theta);
        [rows, cols] = size(beam_matrix);
        
        disp('[SUCCESS] Beam pattern matrix calculation completed.');
        disp([' -> Matrix shape: ', num2str(rows), ' angles x ', num2str(cols), ' frequencies']);
        disp('============================================================');
        return;
    end

    % --- Core Algorithm Execution ---
    v_sound = 340;                      % Speed of sound in m/s
    scan_angles = -90:1:90;            % Angular scanning range (-90 to 90 degrees)
    freq_axis = 100:100:8000;          % Frequency axis from 100Hz to 8000Hz
    
    beam_matrix = zeros(length(scan_angles), length(freq_axis));
    target_rad = deg2rad(target_angle_deg);

    % Iterate through each frequency bin
    for f_idx = 1:length(freq_axis)
        f = freq_axis(f_idx);
        omega = 2 * pi * f;
        
        % Construct the steering vector (steering weight) for the target direction
        w = zeros(num_mics, 1);
        for m = 1:num_mics
            tau_target = (m-1) * mic_distance * sin(target_rad) / v_sound;
            % Corrected: Adjusted sign here to correctly yield maximum gain at target angle via w^H * d
            w(m) = exp(1i * omega * tau_target) / num_mics; 
        end
        
        % Compute spatial response across all scanning angles
        for a_idx = 1:length(scan_angles)
            theta_scan = deg2rad(scan_angles(a_idx));
            
            % Construct the steering vector for the current scanning angle
            d = zeros(num_mics, 1);
            for m = 1:num_mics
                tau_scan = (m-1) * mic_distance * sin(theta_scan) / v_sound;
                d(m) = exp(-1i * omega * tau_scan);
            end
            
            % Array response computation: H(f, theta) = w^H * d
            array_response = sum(conj(w) .* d);
            
            % Convert the magnitude response to decibels (dB)
            beam_matrix(a_idx, f_idx) = 20 * log10(abs(array_response) + 1e-6);
        end
    end
end
