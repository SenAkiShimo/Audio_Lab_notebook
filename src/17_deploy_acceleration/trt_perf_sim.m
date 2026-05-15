function [throughput, memory_footprint] = trt_perf_sim(num_layers, base_flops, precision_mode)
    % Self-test execution driver block
    if nargin == 0
        clc;
        disp('============================================================');
        disp('[RUN] Activating MATLAB TensorRT operator fusion and quantization accelerator simulation...');
        
        % Simulate an ASR model with 12 Transformer encoder layers and 5G FLOPS base calculation
        layers = 12;
        flops = 5e9;
        
        % Simulate performance metrics using FP16 lower precision and 4-stage operator fusion
        [throughput, mem] = trt_perf_sim(layers, flops, 'FP16');
        
        disp('[SUCCESS] Hardware inference performance metrics evaluation complete.');
        disp([' -> Optimized Throughput: ', num2str(throughput), ' FPS']);
        disp([' -> Hard Compressed Memory Footprint: ', num2str(mem), ' MB']);
        disp('============================================================');
        return;
    end

    % ─── Core Algorithm ───
    
    % Establish precision factors based on selected quantization mode
    if strcmp(precision_mode, 'FP32')
        precision_bit = 32;
        speed_multiplier = 1.0;
    elseif strcmp(precision_mode, 'FP16')
        precision_bit = 16;
        speed_multiplier = 2.4; % FP16 activates Tensor Core hardware acceleration
    elseif strcmp(precision_mode, 'INT8')
        precision_bit = 8;
        speed_multiplier = 4.1; % INT8 provides maximal throughput burst
    else
        precision_bit = 32;
        speed_multiplier = 1.0;
    end

    % Simulate layer fusion gains eliminating intermediate VRAM read/write overhead
    fusion_gain_factor = 1 + log10(num_layers);

    % Compute final theoretical throughput and memory footprint metrics
    throughput = (base_flops / 1e6) * speed_multiplier * fusion_gain_factor;
    memory_footprint = (num_layers * 128 * precision_bit) / 8;
end
