#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>

std::vector<float> ResampleINT16ToFloat32(const std::vector<int16_t>& input_pcm16, uint32_t source_sr, uint32_t target_sr) {
    if (source_sr == target_sr) {
        std::vector<float> normalized(input_pcm16.size());
        const float scale = 1.0f / 32768.0f;
        for (size_t i = 0; i < input_pcm16.size(); ++i) {
            normalized[i] = static_cast<float>(input_pcm16[i]) * scale;
        }
        return normalized;
    }

    double resampling_factor = static_cast<double>(target_sr) / source_sr;
    size_t target_size = static_cast<size_t>(std::floor(input_pcm16.size() * resampling_factor));
    std::vector<float> output_buffer(target_size);
    const float int16_to_float_scale = 1.0f / 32768.0f;

    for (size_t n = 0; n < target_size; ++n) {
        double src_float_index = n / resampling_factor;
        size_t src_left_index = static_cast<size_t>(std::floor(src_float_index));
        size_t src_right_index = src_left_index + 1;

        if (src_right_index >= input_pcm16.size()) src_right_index = input_pcm16.size() - 1;

        float weight_right = static_cast<float>(src_float_index - src_left_index);
        float weight_left = 1.0f - weight_right;

        float interpolated_val = weight_left * input_pcm16[src_left_index] + weight_right * input_pcm16[src_right_index];
        output_buffer[n] = interpolated_val * int16_to_float_scale;
    }
    return output_buffer;
}

int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << "[RUN]" << std::endl;

    // Simulate a raw 16-bit signed integer PCM stream (1,000 sample points) with a high sampling rate of 44,100 Hz, as if captured directly by mobile phone microphone hardware.
    std::vector<int16_t> mock_hardware_pcm(1000, 15000);

    // Call the algorithm to forcefully resample and refactor to the 16000Hz ASR standard format for AI voice models.
    std::vector<float> final_float_audio = ResampleINT16ToFloat32(mock_hardware_pcm, 44100, 16000);

    std::cout << "[SUCCESS] Raw Hardware Frame Rate: " << mock_hardware_pcm.size() 
              << " ➔ AI Feature Frame Count After Refactoring and Alignment: " << final_float_audio.size() << std::endl;
    std::cout << "============================================================" << std::endl;
    return 0;
}
