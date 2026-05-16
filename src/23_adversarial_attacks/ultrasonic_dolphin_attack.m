function [demodulated_signal, modulated_ultrasonic] = ultrasonic_dolphin_attack(fs, carrier_fc, duration)
    % Self-test driver configuration
    if nargin == 0
        clc;
        disp('============================================================');
        disp('[RUN] Testing Ultrasonic DolphinAttack Carrier Modulation and Demodulation...');
        
        % High sampling rate (96kHz) to avoid aliasing for a 25kHz ultrasonic carrier
        fs_high = 96000;
        fc_ultrasonic = 25000; 
        t_duration = 0.1;
        
        [demodulated_signal, modulated_ultrasonic] = ultrasonic_dolphin_attack(fs_high, fc_ultrasonic, t_duration);
        
        disp('[SUCCESS] Hardware nonlinear physical demodulation simulation completed.');
        disp([' -> Transmitted ultrasonic buffer size: ', num2str(length(modulated_ultrasonic)), ' samples']);
        disp([' -> Demodulated hardware baseband buffer size: ', num2str(length(demodulated_signal)), ' samples']);
        disp('============================================================');
        return;
    end

    % --- Core Algorithm Execution ---
    t = 0:(1/fs):duration;
    
    % 1. Generate low-frequency baseband command audio (e.g., 400Hz pure tone)
    f_command = 400;
    m_t = sin(2 * pi * f_command * t);
    
    % 2. Apply standard Double-Sideband Amplitude Modulation (AM)
    % Formula: s(t) = [1 + k * m(t)] * cos(2 * pi * fc * t)
    modulation_index = 0.7; 
    carrier_wave = cos(2 * pi * carrier_fc * t);
    modulated_ultrasonic = (1 + modulation_index * m_t) .* carrier_wave;
    
    % 3. Simulate free-space attenuation during acoustic propagation
    attenuation_factor = 0.15;
    received_at_mic = modulated_ultrasonic * attenuation_factor;
    
    % 4. Simulate the second-order square-law nonlinearity of the microphone amplifier
    % Formula: V_out = a1 * V_in + a2 * V_in^2
    % The V_in^2 component produces a difference frequency (fc - fc) that recovers the baseband signal
    a1 = 1.0;
    a2 = 0.5; 
    hardware_electric_signal = a1 * received_at_mic + a2 * (received_at_mic .^ 2);
    
    % 5. Apply a 4th-order Butterworth low-pass filter to extract the demodulated baseband signal
    [b, a] = butter(4, 2000 / (fs/2), 'low');
    demodulated_signal = filter(b, a, hardware_electric_signal);
end
