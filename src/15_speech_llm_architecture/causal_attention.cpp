// C++ Source: High-Performance Multimodal Causal Self-Attention Layer & Full-Duplex Interruption State Machine
// Author: Yang

#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <cstdio>

class CausalAttentionEngine {
public:
    // Computes causal self-attention with a mask for multimodal sequences.
    // Simulates non-linear interactions of mixed text and audio vectors within a Transformer.
    static bool ComputeCausalAttentionWithInterruption(
        const std::vector<std::vector<float>>& unified_sequence, 
        size_t seq_len, 
        size_t hidden_dim, 
        const std::vector<bool>& mock_user_interruption_stream) {
        
        std::cout << "[ATTENTION_BLOCK] Starting causal autoregressive deduction for mixed sequence of length " << seq_len << "..." << std::endl;

        // Full-duplex interruption gating loop
        for (size_t t = 0; t < seq_len; ++t) {
            // 1. Hardware-level simulation: check asynchronous upstream network for user interruption signals
            if (mock_user_interruption_stream[t]) {
                std::cout << "[HARD_INTERRUPT] Warning: Intercepted strong full-duplex user interruption at step t=" << t << "!" << std::endl;
                std::cout << " -> Action: Reversing state machine flow immediately. Clearing current forward computation tree and appending <|stop|> token." << std::endl;
                std::cout << "[STATUS] Current autoregressive inference aborted. Switching system to listening mode." << std::endl;
                return false; // Trigger hard interrupt and exit early
            }

            // 2. Simulate causal self-attention at step t: node can only attend to history (j <= t)
            float attention_energy_accumulator = 0.0f;
            for (size_t j = 0; j <= t; ++j) {
                // Compute dot-product between Query and Key vectors
                float dot_product = 0.0f;
                for (size_t d = 0; d < hidden_dim; ++d) {
                    dot_product += unified_sequence[t][d] * unified_sequence[j][d];
                }
                // Apply scaling factor and accumulate
                attention_energy_accumulator += std::exp(dot_product / std::sqrt(static_cast<float>(hidden_dim)));
            }

            // Print running status every 40 steps to prevent console flooding
            if (t % 40 == 0) {
                std::printf(" -> Frame step t=%03zu: Causal attention energy converged. Current hidden score: %.2f\n", t, attention_energy_accumulator);
            }
        }

        std::cout << "[SUCCESS] Full sequence causal self-attention completed without interruption." << std::endl;
        return true;
    }
};

int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << "[RUN] Activating C++ multimodal causal attention and full-duplex interruption state machine testing..." << std::endl;

    size_t sequence_length = 156; // Mixed sequence composed of 6 text tokens and 150 audio frames
    size_t hidden_dimension = 64; // Reduced hidden dimension for simplified testing

    // 1. Construct high-dimensional multimodal hidden sequence matrix [156, 64]
    std::vector<std::vector<float>> mock_unified_matrix(sequence_length, std::vector<float>(hidden_dimension, 0.1f));

    // 2. Create an asynchronous full-duplex event stream: user starts speaking at t = 120 (triggering interruption)
    std::vector<bool> mock_interruption_event(sequence_length, false);
    mock_interruption_event[120] = true;

    // 3. Feed sequence into attention kernel and observe state machine interruption behavior
    bool is_completed = CausalAttentionEngine::ComputeCausalAttentionWithInterruption(
        mock_unified_matrix, sequence_length, hidden_dimension, mock_interruption_event);

    std::cout << "============================================================" << std::endl;
    return 0;
}
