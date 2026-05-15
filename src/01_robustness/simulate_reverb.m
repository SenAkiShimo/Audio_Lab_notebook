function [reverberant_signal, rir] = simulate_reverb(clean_signal, fs, room_dim, src_pos, mic_pos)
    % Check if there are no input parameters; this indicates that the user directly clicked "Run," in which case the program automatically enters the self-test code block.
    if nargin == 0
        clc; disp('============================================================');
        disp('[RUN]');
        
        % Synthesized Input Data: A 4-second, 16 kHz sampled, pristine mono signal.
        fs_test = 16000;
        clean_signal_test = randn(fs_test * 2, 1) * 0.1; 
        room_dim_test = [5, 4, 3];  % A 5x4x3-meter room
        src_pos_test = [1, 1, 1.5]; % A 1.5-meter-high speaker position
        mic_pos_test = [3, 2, 1.5]; % A 1.5-meter-high microphone position
        
        % Call itself to perform the RIR convolution and generate the reverberant signal
        [reverberant_signal, rir] = simulate_reverb(clean_signal_test, fs_test, room_dim_test, src_pos_test, mic_pos_test);
        
        disp(['[SUCCESS] RIR filter length: ', num2str(length(rir)), ' taps']);
        disp('============================================================');
        return;
    end

    c = 340; rt60 = 0.4; 
    num_samples_rir = round(rt60 * fs); rir = zeros(num_samples_rir, 1);
    dist_direct = norm(src_pos - mic_pos); time_direct = dist_direct / c;
    sample_direct = round(time_direct * fs) + 1;
    if sample_direct <= num_samples_rir, rir(sample_direct) = 1.0 / dist_direct; end
    reflection_coeff = 0.65; walls = [0, room_dim(1); 0, room_dim(2); 0, room_dim(3)];
    for w = 1:3
        for side = 1:2
            virtual_src = src_pos; virtual_src(w) = 2 * walls(w, side) - src_pos(w);
            dist_reflect = norm(virtual_src - mic_pos); time_reflect = dist_reflect / c;
            sample_reflect = round(time_reflect * fs) + 1;
            if sample_reflect <= num_samples_rir
                rir(sample_reflect) = rir(sample_reflect) + (reflection_coeff / dist_reflect);
            end
        end
    end
    reverberant_signal = conv(clean_signal, rir);
    reverberant_signal = reverberant_signal / max(abs(reverberant_signal));
end