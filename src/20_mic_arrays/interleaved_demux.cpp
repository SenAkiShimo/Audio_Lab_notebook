#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdio>

class AudioHardwareDemuxer {
public:
    // Demultiplexes interleaved 16-bit PCM binary data into a multi-channel float matrix.
    static std::vector<std::vector<float>> ExecuteChannelDemux(
        const std::vector<int16_t>& interleaved_pcm16, 
        size_t num_channels) 
    {
        size_t total_samples = interleaved_pcm16.size();
        size_t samples_per_channel = total_samples / num_channels;

        // Allocate continuous memory for multi-channel output matrix [num_channels, samples_per_channel]
        std::vector<std::vector<float>> multi_channel_matrix(num_channels, std::vector<float>(samples_per_channel));
        const float int16_to_float_scale = 1.0f / 32768.0f;

        // Process demultiplexing frame by frame
        // Interleaved layout: [Mic0_T0, Mic1_T0, Mic2_T0, Mic0_T1, Mic1_T1...]
        for (size_t i = 0; i < samples_per_channel; ++i) {
            size_t base_interleaved_index = i * num_channels;
            
            for (size_t ch = 0; ch < num_channels; ++ch) {
                // Fetch raw sample from the interleaved frame
                int16_t raw_sample = interleaved_pcm16[base_interleaved_index + ch];
                
                // Convert type to Float32, normalize to [-1.0, 1.0], and store into specific channel buffer
                multi_channel_matrix[ch][i] = static_cast<float>(raw_sample) * int16_to_float_scale;
            }
        }

        return multi_channel_matrix;
    }
};

// Test driver
int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << "[RUN] Testing Multi-Channel Interleaved PCM Demuxer..." << std::endl;

    size_t test_channels = 4;          // Simulating a 4-microphone array
    size_t test_frames_per_ch = 100;   // 100 samples per channel
    size_t total_buffer_size = test_channels * test_frames_per_ch;

    // 1. Generate a mock interleaved PCM buffer
    std::vector<int16_t> mock_dma_buffer(total_buffer_size);
    for (size_t i = 0; i < total_buffer_size; ++i) {
        mock_dma_buffer[i] = static_cast<int16_t>((i % test_channels) * 5000 + 2000);
    }

    // 2. Perform channel demultiplexing
    std::vector<std::vector<float>> output_matrix = AudioHardwareDemuxer::ExecuteChannelDemux(mock_dma_buffer, test_channels);

    std::cout << "[SUCCESS] Audio demultiplexing completed." << std::endl;
    std::cout << " -> Input size: " << mock_dma_buffer.size() * sizeof(int16_t) << " bytes" << std::endl;
    std::cout << " -> Extracted channels: " << output_matrix.size() << " channels" << std::endl;
    if (!output_matrix.empty()) {
        std::cout << " -> Output samples per channel: " << output_matrix[0].size() << " frames (Float32)" << std::endl;
    }
    std::cout << "============================================================" << std::endl;

    return 0;
}
