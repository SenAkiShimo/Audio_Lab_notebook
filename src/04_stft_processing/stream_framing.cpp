#include <iostream>
#include <vector>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class AudioStreamFramingEngine {
public:
    // Generates a standard Hann window array
    static std::vector<float> GenerateHannWindow(size_t frame_length) {
        std::vector<float> window(frame_length);
        
        // FIX 1: Boundary protection against division by zero if frame_length is 1 or less
        if (frame_length <= 1) {
            if (frame_length == 1) window[0] = 1.0f;
            return window;
        }

        for (size_t i = 0; i < frame_length; ++i) {
            // Hann window formula: 0.5 * (1 - cos(2 * pi * i / (N - 1)))
            window[i] = 0.5f * (1.0f - std::cos(2.0f * static_cast<float>(M_PI) * i / (frame_length - 1)));
        }
        return window;
    }

    // Segments input signal into overlapping frames and applies the window function
    static std::vector<std::vector<float>> ExecuteFramingAndWindowing(
        const std::vector<float>& input_signal, size_t frame_length, size_t hop_length) {
        
        std::vector<std::vector<float>> framed_matrix;

        // FIX 2: Defensive check to prevent infinite loops or invalid configurations
        if (frame_length == 0 || hop_length == 0 || input_signal.size() < frame_length) {
            return framed_matrix;
        }

        // Generate the window function once to avoid redundant computations inside the loop
        std::vector<float> hann_window = GenerateHannWindow(frame_length);

        // Compute the total number of valid frames
        size_t num_frames = (input_signal.size() - frame_length) / hop_length + 1;
        framed_matrix.reserve(num_frames);

        // Slide the window across the time domain
        for (size_t f = 0; f < num_frames; ++f) {
            size_t start_ptr = f * hop_length;
            std::vector<float> current_frame(frame_length);

            for (size_t i = 0; i < frame_length; ++i) {
                // Fetch the raw sample within the sliding window
                float raw_sample = input_signal[start_ptr + i];
                
                // Apply the window coefficient to mitigate spectral leakage
                current_frame[i] = raw_sample * hann_window[i];
            }
            framed_matrix.push_back(current_frame);
        }
        return framed_matrix;
    }
};

int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << "[RUN] Initializing C++ time-domain streaming framing engine test driver..." << std::endl;

    // 1. Simulate a 1-second audio sequence (16000 samples) at a 16 kHz sampling rate
    std::vector<float> mock_audio_stream(16000, 0.5f);

    // 2. Set standard ASR pre-processing parameters: 400 samples (25ms), hop size 160 samples (10ms)
    size_t frame_len = 400;
    size_t hop_len = 160;

    // 3. Execute framing and windowing operations
    std::vector<std::vector<float>> result_matrix = AudioStreamFramingEngine::ExecuteFramingAndWindowing(mock_audio_stream, frame_len, hop_len);

    std::cout << "[SUCCESS] Time-domain signal framing completed." << std::endl;
    std::cout << " -> Original audio length: " << mock_audio_stream.size() << " samples" << std::endl;
    std::cout << " -> Total segmented frames: " << result_matrix.size() << " frames" << std::endl;
    if (!result_matrix.empty()) {
        std::cout << " -> Dynamic frame width: " << result_matrix[0].size() << " samples (Hann window applied)" << std::endl;
    }
    std::cout << "============================================================" << std::endl;

    return 0;
}
