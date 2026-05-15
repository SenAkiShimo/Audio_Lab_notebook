#include <iostream>
#include <vector>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class SincResampler {
private:
    // Computes the normalized sinc function value: sinc(x) = sin(pi * x) / (pi * x)
    static double CalculateSinc(double x) {
        if (std::abs(x) < 1e-9) return 1.0;
        return std::sin(M_PI * x) / (M_PI * x);
    }

    // Computes the Hann window coefficient to mitigate the Gibbs phenomenon caused by time-domain truncation
    static double CalculateHannWindow(double i, double window_size) {
        return 0.5 * (1.0 + std::cos(2.0 * M_PI * i / window_size));
    }

public:
    static std::vector<float> ComputeSincResample(const std::vector<float>& input_signal, double source_sr, double target_sr) {
        double factor = target_sr / source_sr;
        size_t target_size = static_cast<size_t>(std::floor(input_signal.size() * factor));
        std::vector<float> output_signal(target_size, 0.0f);

        // Set the Whittaker-Shannon interpolation truncation semi-window size
        int l_semi_window = 12;

        // Perform time-domain sinc convolution interpolation
        for (size_t n = 0; n < target_size; ++n) {
            double src_float_time = n / factor;
            size_t center_idx = static_cast<size_t>(std::floor(src_float_time));
            double sum_weight = 0.0;
            double sum_value = 0.0;

            // Apply bidirectional windowed convolution around the current time mapping point
            for (int i = -l_semi_window; i <= l_semi_window; ++i) {
                int current_src_idx = static_cast<int>(center_idx) + i;

                // Boundary protection check
                if (current_src_idx >= 0 && current_src_idx < static_cast<int>(input_signal.size())) {
                    double time_offset = src_float_time - current_src_idx;
                    
                    // Compute low-pass interpolation weight with a window function
                    double weight = CalculateSinc(time_offset) * CalculateHannWindow(i, l_semi_window * 2);
                    sum_value += input_signal[current_src_idx] * weight;
                    sum_weight += weight;
                }
            }

            // Normalize energy output
            output_signal[n] = (sum_weight > 0.0) ? static_cast<float>(sum_value / sum_weight) : 0.0f;
        }
        return output_signal;
    }
};

int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << "[RUN] Initializing C++ time-domain sinc resampler test driver..." << std::endl;

    // 1. Generate a mock floating-point signal sampled at 16000 Hz
    std::vector<float> mock_input(400, 0.0f);
    for (size_t i = 0; i < mock_input.size(); ++i) {
        mock_input[i] = std::sin(2.0 * M_PI * 1000.0 * i / 16000.0); // 1000 Hz pure tone
    }

    // 2. Perform resampling conversion from 16000 Hz to 24000 Hz using sinc interpolation
    std::vector<float> resampled_output = SincResampler::ComputeSincResample(mock_input, 16000, 24000);

    std::cout << "[SUCCESS] Resampling computation completed." << std::endl;
    std::cout << " -> Original signal length: " << mock_input.size() << " samples" << std::endl;
    std::cout << " -> Resampled signal length: " << resampled_output.size() << " samples" << std::endl;
    std::cout << "============================================================" << std::endl;

    return 0;
}
