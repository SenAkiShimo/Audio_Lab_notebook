#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstdio>

class KwsStreamingPostProcessor {
private:
    float high_threshold_;         // Upper rigid threshold to initiate a trigger
    float low_threshold_;          // Lower flexible threshold to maintain an active tracking state
    size_t smooth_window_size_;    // Sliding time window duration represented in frames
    std::vector<float> ring_buffer_; // Memory-optimized ring buffer acting as a sliding window
    size_t write_ptr_;             // Sequential index pointer for cyclic ring buffer writing
    bool is_activated_;            // Latched internal tracking state of the keyword spotter

public:
    KwsStreamingPostProcessor(float high_threshold = 0.85f, float low_threshold = 0.40f, size_t window_size = 5)
        : high_threshold_(high_threshold), low_threshold_(low_threshold), smooth_window_size_(window_size), write_ptr_(0), is_activated_(false) {
        
        // FIX 1: Defensive prevention against division-by-zero bounds errors
        if (smooth_window_size_ == 0) {
            smooth_window_size_ = 1;
        }
        ring_buffer_.resize(smooth_window_size_, 0.0f);
    }

    // Processes the raw frame confidence probability via moving average smoothing and dual-threshold state machine tracking
    bool ProcessFrameProbability(float raw_prob, float& out_smoothed_prob) {
        // 1. Enqueue new feature probability into the ring buffer, overwriting the oldest entry
        ring_buffer_[write_ptr_] = raw_prob;
        write_ptr_ = (write_ptr_ + 1) % smooth_window_size_;

        // 2. Compute the moving average across the current window capacity to suppress transient noise spikes
        float sum = std::accumulate(ring_buffer_.begin(), ring_buffer_.end(), 0.0f);
        out_smoothed_prob = sum / static_cast<float>(smooth_window_size_);

        bool is_wake_word_triggered = false;

        // 3. FIX 2: Implementation of a proper dual-threshold flow control state machine
        if (!is_activated_) {
            // Trigger activation only when the smoothed energy clears the rigid high threshold bound
            if (out_smoothed_prob >= high_threshold_) {
                is_activated_ = true;
            }
        } else {
            // Deactivate tracking if the continuous energy falls below the low threshold baseline
            if (out_smoothed_prob < low_threshold_) {
                is_activated_ = false;
            }
        }

        // 4. Secondary verification pass using maximum historical peak validation
        if (is_activated_) {
            float max_historical = *std::max_element(ring_buffer_.begin(), ring_buffer_.end());
            if (max_historical >= high_threshold_) {
                is_wake_word_triggered = true;
                
                // Clear trajectory tracking state post-triggering to prevent redundant back-to-back alerts
                std::fill(ring_buffer_.begin(), ring_buffer_.end(), 0.0f);
                write_ptr_ = 0;
                is_activated_ = false;
            }
        }

        return is_wake_word_triggered;
    }
};

int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << "[RUN] Activating C++ edge streaming KWS smoothing filter and state machine test driver..." << std::endl;

    // Instantiate post-processor: high_threshold = 0.85, low_threshold = 0.40, window_size = 5 frames
    KwsStreamingPostProcessor kws_guard(0.85f, 0.40f, 5);

    // Simulate 20 frames of runtime inference telemetry data
    // Index t=10 presents an isolated transient distortion spike (0.92) caused by environmental noise
    // Indices t=14 to t=18 represent a continuous and stable keyword vocal articulation event
    std::vector<float> mock_inference_stream = {
        0.02f, 0.05f, 0.01f, 0.03f, 0.02f, 0.04f, 0.01f, 0.02f, 0.05f, 0.02f, 
        0.92f, // t=10: Isolated noise spike
        0.12f, 0.05f, 
        0.86f, 0.89f, 0.91f, 0.88f, 0.85f, // t=14~18: True continuous keyword duration
        0.22f, 0.04f
    };

    for (size_t t = 0; t < mock_inference_stream.size(); ++t) {
        float raw_p = mock_inference_stream[t];
        float smoothed_p = 0.0f;
        bool triggered = kws_guard.ProcessFrameProbability(raw_p, smoothed_p);
        
        std::printf("Time step t=%02zu | Raw output: %.2f | Smoothed average: %.2f | Interrupt status: %s\n", 
                    t, raw_p, smoothed_p, triggered ? "[TRIGGER_WAKE]" : "[WAITING]");
    }
    std::cout << "============================================================" << std::endl;
    return 0;
}
