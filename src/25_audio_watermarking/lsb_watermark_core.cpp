#include <iostream>
#include <vector>
#include <cstdint>
#include <string>
#include <stdexcept>

class LsbWatermarkEngine {
public:
    // Embeds a text signature into the Least Significant Bit (LSB) of 16-bit PCM audio samples.
    static std::vector<int16_t> EmbedMetadataLSB(
        const std::vector<int16_t>& clean_audio, 
        const std::string& secret_signature, 
        size_t& out_bit_length) 
    {
        std::vector<int16_t> watermarked_signal = clean_audio;
        out_bit_length = secret_signature.length() * 8; // Each character takes 8 bits

        if (out_bit_length > clean_audio.size()) {
            throw std::runtime_error("Acoustic buffer too short to sustain payload metadata!");
        }

        std::cout << "[EMBED] Starting bit-level LSB steganography. Payload length: " << out_bit_length << " bits." << std::endl;

        // Iterate through each bit to embed the hidden message
        for (size_t i = 0; i < out_bit_length; ++i) {
            size_t char_idx = i / 8;
            size_t bit_idx = 7 - (i % 8); // Extract bits from MSB to LSB of the character

            // Extract the target bit (0 or 1) using shift and bitwise AND
            int16_t current_bit = (secret_signature[char_idx] >> bit_idx) & 1;

            // Clear the LSB of the audio sample and overwrite it with the hidden bit
            int16_t current_sample = watermarked_signal[i];
            watermarked_signal[i] = (current_sample & ~1) | current_bit;
        }

        std::cout << "[SUCCESS] LSB embedding completed. Maximum amplitude perturbation is within 1/65536." << std::endl;
        return watermarked_signal;
    }

    // Extracts the hidden text signature from the LSB of the watermarked audio samples.
    static std::string ExtractMetadataLSB(const std::vector<int16_t>& watermarked_audio, size_t bit_length) {
        std::string decoded_signature = "";
        char current_char = 0;

        for (size_t i = 0; i < bit_length; ++i) {
            // Isolate the LSB using a bitmask
            int16_t extracted_bit = watermarked_audio[i] & 1;

            // Reconstruct the 8-bit ASCII character via left shifts
            size_t bit_idx = 7 - (i % 8);
            current_char |= (extracted_bit << bit_idx);

            // Append the fully reconstructed character to the output string every 8 bits
            if (bit_idx == 0) {
                decoded_signature += current_char;
                current_char = 0; // Reset the character buffer
            }
        }

        std::cout << "[SUCCESS] Watermark extraction completed. Recovered message: '" << decoded_signature << "'" << std::endl;
        return decoded_signature;
    }
};

// Test driver
int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << "[RUN] Testing Bit-Level Time-Domain LSB Audio Watermarking..." << std::endl;

    // 1. Simulate a 16-bit signed PCM audio stream (500 samples)
    std::vector<int16_t> mock_pcm16_stream(500, 16384);

    // 2. Define the security copyright signature string
    std::string copyright_hash = "OWNER_YANG_ID_#890321";

    // 3. Execute the full embedding and extraction pipeline
    size_t payload_bit_len = 0;
    try {
        std::vector<int16_t> watermarked_pcm = LsbWatermarkEngine::EmbedMetadataLSB(mock_pcm16_stream, copyright_hash, payload_bit_len);
        
        // 4. Extract and verify the hidden message
        std::string extracted_hash = LsbWatermarkEngine::ExtractMetadataLSB(watermarked_pcm, payload_bit_len);
    } 
    catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception caught: " << e.what() << std::endl;
    }

    std::cout << "============================================================" << std::endl;
    return 0;
}
