#include <iostream>
#include <vector>
#include <cstdint>

class PreEmphasisFilter {
public:
    // Block-based processing edition: Executes 1st-order high-pass differential filtering.
    // Supports continuous streaming by passing an optional pointer to maintain historical state across frames.
    static std::vector<float> ExecutePreEmphasis(const std::vector<float>& input_signal, float alpha = 0.97f, float* prev_sample = nullptr) {
        std::vector<float> output_signal(input_signal.size());
        if (input_signal.empty()) return output_signal;

        size_t start_idx = 0;

        // If a valid streaming state pointer is provided, use the last sample from the previous block
        if (prev_sample != nullptr) {
            output_signal[0] = input_signal[0] - alpha * (*prev_sample);
            start_idx = 1;
        } else {
            // Monolithic chunk fallback: The very first sample preserves its own value due to zero historical state
            output_signal[0] = input_signal[0];
            start_idx = 1;
        }

        // Apply sequential time-domain finite impulse response (FIR) difference calculation
        for (size_t t = start_idx; t < input_signal.size(); ++t) {
            output_signal[t] = input_signal[t] - alpha * input_signal[t - 1];
        }

        // Update the streaming history tracker for the next overlapping block segment
        if (prev_sample != nullptr) {
            *prev_sample = input_signal.back();
        }

        return output_signal;
    }
};

int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << "[RUN] Initializing C++ time-domain 1st-order pre-emphasis high-pass filter test driver..." << std::endl;

    // 1. Simulate a 1-second continuous audio sequence (16000 samples) at a 16 kHz sampling rate
    std::vector<float> mock_pcm_stream(16000, 0.8f);

    // 2. Execute the pre-emphasis filtering using the standard coefficient alpha = 0.97
    std::vector<float> filtered_stream = PreEmphasisFilter::ExecutePreEmphasis(mock_pcm_stream, 0.97f);

    std::cout << "[SUCCESS] High-frequency energy differential compensation completed." << std::endl;
    std::cout << " -> Original buffer length: " << mock_pcm_stream.size() << " samples" << std::endl;
    std::cout << " -> Filtered buffer length: " << filtered_stream.size() << " samples" << std::endl;
    std::cout << "============================================================" << std::endl;

    return 0;
}
