#include <iostream>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cstdio> // Required for std::remove

#pragma pack(push, 1)
struct WavHeaderStructure {
    char riff_id[4];           // Bytes 1-4: Must be "RIFF"
    uint32_t riff_size;        // Bytes 5-8: Total file size minus 8 bytes
    char wave_fmt[4];          // Bytes 9-12: Must be "WAVE"
    char fmt_id[4];            // Bytes 13-16: Must be "fmt "
    uint32_t fmt_size;         // Bytes 17-20: Size of format chunk (PCM = 16)
    uint16_t audio_format;     // Bytes 21-22: Audio format (PCM = 1)
    uint16_t num_channels;     // Bytes 23-24: Number of channels
    uint32_t sample_rate;      // Bytes 25-28: Sampling rate (Hz)
    uint32_t byte_rate;        // Bytes 29-32: Byte rate (sample_rate * block_align)
    uint16_t block_align;      // Bytes 33-34: Block align (num_channels * bits_per_sample / 8)
    uint16_t bits_per_sample;  // Bytes 35-36: Bit depth (e.g., 16 bits)
    char data_id[4];           // Bytes 37-40: Must be "data"
    uint32_t data_size;        // Bytes 41-44: Total bytes of raw PCM audio data
};
#pragma pack(pop)

class WavHeaderParser {
public:
    static bool ParseWavHeaderSpec(const std::string& filepath, WavHeaderStructure& out_header) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "[ERROR] Failed to open target audio file: " << filepath << std::endl;
            return false;
        }

        // FIX 1: Explicitly clear target buffer to prevent memory pollution
        std::memset(&out_header, 0, sizeof(WavHeaderStructure));

        // Read exactly 44 bytes from the file into the binary structure
        file.read(reinterpret_cast<char*>(&out_header), sizeof(WavHeaderStructure));

        // FIX 2: Fail-Fast if the file is truncated or smaller than a standard WAV header
        if (file.gcount() < static_cast<std::streamsize>(sizeof(WavHeaderStructure))) {
            std::cerr << "[SECURITY_ALERT] Malformed header. File size is insufficient." << std::endl;
            return false;
        }

        // Validate the RIFF and WAVE magic numbers to prevent container hijacking
        if (std::strncmp(out_header.riff_id, "RIFF", 4) != 0 || std::strncmp(out_header.wave_fmt, "WAVE", 4) != 0) {
            std::cerr << "[SECURITY_ALERT] Non-standard RIFF/WAV container. Parsing intercepted." << std::endl;
            return false;
        }

        // FIX 3: Explicitly close the handle to release OS file locks immediately
        file.close(); 
        return true;
    }
};

int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << "[RUN] Activating C++ Hardware-Level Binary WAV Parser Test Driver..." << std::endl;

    // 1. Forge a compliant 44-byte standard lossless WAV header block in memory
    WavHeaderStructure mock_header;
    std::memcpy(mock_header.riff_id, "RIFF", 4);
    mock_header.riff_size = 32044;
    std::memcpy(mock_header.wave_fmt, "WAVE", 4);
    std::memcpy(mock_header.fmt_id, "fmt ", 4);
    mock_header.fmt_size = 16;
    mock_header.audio_format = 1; // 1 = Raw Linear PCM
    mock_header.num_channels = 1; // Mono channel
    mock_header.sample_rate = 16000; // 16kHz sampling rate
    mock_header.byte_rate = 32000;
    mock_header.block_align = 2;
    mock_header.bits_per_sample = 16; // 16-bit depth
    std::memcpy(mock_header.data_id, "data", 4);
    mock_header.data_size = 32000; // Exactly 1 second of raw PCM data volume

    // 2. Write the memory block to a local temporary file to simulate actual disk I/O
    std::string test_filename = "mock_test_audio.wav";
    std::ofstream out_file(test_filename, std::ios::binary);
    if (!out_file) {
        std::cerr << "[ERROR] Failed to create mock test file." << std::endl;
        return 1;
    }
    out_file.write(reinterpret_cast<char*>(&mock_header), sizeof(WavHeaderStructure));
    out_file.close();

    // 3. Invoke the parser to decode and verify the binary metadata structure
    WavHeaderStructure parsed_result;
    if (WavHeaderParser::ParseWavHeaderSpec(test_filename, parsed_result)) {
        std::cout << "[SUCCESS] Binary audio metadata parsed successfully!" << std::endl;
        std::cout << " -> Channels: " << parsed_result.num_channels << std::endl;
        std::cout << " -> Sample Rate: " << parsed_result.sample_rate << " Hz" << std::endl;
        std::cout << " -> Bit Depth: " << parsed_result.bits_per_sample << " bits" << std::endl;
        std::cout << " -> Raw PCM Data Size: " << parsed_result.data_size << " bytes" << std::endl;
    }

    // Clean up the temporary file from the disk environment
    std::remove(test_filename.c_str());
    std::cout << "============================================================" << std::endl;
    return 0;
}
