#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <cstdio>

class FractionalDelaySumBeamformer {
public:
    // Performs fractional delay alignment and multichannel delay-and-sum beamforming.
    static std::vector<float> ExecuteBeamforming(
        const std::vector<std::vector<float>>& multichannel_inputs,
        const std::vector<double>& tdoa_per_channel, 
        double fs) 
    {
        size_t num_channels = multichannel_inputs.size();
        size_t num_samples = multichannel_inputs[0].size();
        std::vector<float> output_mono(num_samples, 0.0f);

        // Loop through each time sample
        for (size_t t = 0; t < num_samples; ++t) {
            float channel_accumulator = 0.0f;

            for (size_t m = 0; m < num_channels; ++m) {
                // Convert physical TDOA (seconds) to discrete-time delay (samples)
                double sample_offset = tdoa_per_channel[m] * fs;

                // Split delay into integer part (base index) and fractional part (interpolation weight)
                double int_part_floor;
                double frac_part = std::modf(sample_offset, &int_part_floor);
                
                int64_t base_idx = static_cast<int64_t>(t) + static_cast<int64_t>(int_part_floor);

                // First-order linear fractional interpolation
                float interpolated_sample = 0.0f;
                if (base_idx >= 0 && base_idx + 1 < static_cast<int64_t>(num_samples)) {
                    float left_sample = multichannel_inputs[m][base_idx];
                    float right_sample = multichannel_inputs[m][base_idx + 1];
                    interpolated_sample = (1.0f - static_cast<float>(frac_part)) * left_sample + static_cast<float>(frac_part) * right_sample;
                } 
                // Boundary check: fallback to the last valid sample if the next one is out of bounds
                else if (base_idx >= 0 && base_idx < static_cast<int64_t>(num_samples)) {
                    interpolated_sample = multichannel_inputs[m][base_idx];
                }

                channel_accumulator += interpolated_sample;
            }

            // Average the accumulated samples across all channels (Sum & Normalize)
            output_mono[t] = channel_accumulator / num_channels;
        }

        return output_mono;
    }
};

// Test driver
int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << "[RUN] Testing Time-Domain Fractional Delay-and-Sum Beamformer..." << std::endl;

    size_t mics = 4;
    size_t samples = 500;
    std::vector<std::vector<float>> mock_multichannel(mics, std::vector<float>(samples, 0.3f));

    // Simulated TDOA per channel in seconds
    std::vector<double> mock_tdoa = { 0.0, 2.34e-5, 4.68e-5, 7.02e-5 };

    // Execute beamforming with standard sampling rate 16000Hz
    std::vector<float> clean_mono = FractionalDelaySumBeamformer::ExecuteBeamforming(mock_multichannel, mock_tdoa, 16000.0);

    std::cout << "[SUCCESS] Beamforming processing completed." << std::endl;
    std::cout << " -> Input: " << mics << " channels x " << samples << " samples" << std::endl;
    std::cout << " -> Output: " << clean_mono.size() << " samples" << std::endl;
    std::cout << "============================================================" << std::endl;

    return 0;
}
