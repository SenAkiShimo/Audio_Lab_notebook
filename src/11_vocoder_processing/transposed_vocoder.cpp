#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdio>

class TransposedConv1DEngine {
public:
    // Executes a 1D transposed convolution operator using an overlap-add algorithm.
    // Reconstructs low-dimensional acoustic features (e.g., Mel-spectrograms) into time-domain sequences.
    static std::vector<float> ComputeTransposedConv1D(
        const std::vector<float>& input_mels_flat, const std::vector<float>& kernel_weights, size_t stride, size_t kernel_size) {
        
        // FIX 1: Guard against early underflow if the input vector sequence is empty
        if (input_mels_flat.empty() || kernel_weights.empty() || kernel_size == 0) {
            return std::vector<float>();
        }

        // FIX 2: Prevent invalid layout alignment caused by zero stride configuration
        size_t safe_stride = (stride == 0) ? 1 : stride;

        size_t input_len = input_mels_flat.size();
        
        // Compute exact upsampled sequence length: Output_Length = (Input_Length - 1) * Stride + Kernel_Size
        size_t output_len = (input_len - 1) * safe_stride + kernel_size;
        std::vector<float> output_waveform(output_len, 0.0f);

        // Perform spatial-domain overlap-add pipeline conversion
        for (size_t i = 0; i < input_len; ++i) {
            size_t output_start_ptr = i * safe_stride;
            float input_weight = input_mels_flat[i];

            for (size_t k = 0; k < kernel_size; ++k) {
                // Accumulate weights onto the output time-domain buffer
                output_waveform[output_start_ptr + k] += input_weight * kernel_weights[k];
            }
        }
        return output_waveform;
    }
};

int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << "[RUN] Activating C++ neural vocoder transposed convolution upsampling kernel test driver..." << std::endl;

    // 1. Simulate a single discrete channel sequence (100 feature frames) derived from an acoustic model
    std::vector<float> mock_mel_frame_stream(100, -2.5f);

    // 2. Simulate 1D convolutional kernel weights (Kernel Size = 8) from a pre-trained GAN generator
    std::vector<float> pre_trained_kernel = { 0.15f, -0.32f, 0.54f, 0.88f, -0.41f, 0.12f, -0.05f, 0.01f };

    // 3. Set the upsampling factor configuration: Stride = 4 (temporal expansion multiplier)
    size_t stride_factor = 4;
    size_t k_size = pre_trained_kernel.size();

    // 4. Trigger the upsampling convolution process
    std::vector<float> upsampled_waveform = TransposedConv1DEngine::ComputeTransposedConv1D(
        mock_mel_frame_stream, pre_trained_kernel, stride_factor, k_size);

    std::cout << "[SUCCESS] Spatio-temporal sequence upsampling completed successfully." << std::endl;
    std::cout << " -> Input acoustic frames: " << mock_mel_frame_stream.size() << " frames" << std::endl;
    std::cout << " -> Upsampling stride factor: " << stride_factor << "x expansion" << std::endl;
    std::cout << " -> Output waveform samples: " << upsampled_waveform.size() << " elements" << std::endl;
    std::cout << "============================================================" << std::endl;

    return 0;
}
