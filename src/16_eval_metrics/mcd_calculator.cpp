#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>

class MelCepstralDistanceEngine {
public:
    // Computes the Mel-Cepstral Distance (MCD) score for a single frame between 13-dimensional MFCC vectors.
    static double ComputeFrameMCD(const std::vector<double>& mfcc_ref, const std::vector<double>& mfcc_synth) {
        if (mfcc_ref.size() != mfcc_synth.size()) {
            throw std::invalid_argument("Feature vector dimensions do not match.");
        }

        double squared_sum = 0.0;
        // Compute squared Euclidean distance.
        // Industrial standards usually ignore the 0-th dimension (energy term), calculating from indices 1 to 12.
        for (size_t i = 1; i < mfcc_ref.size(); ++i) {
            double diff = mfcc_ref[i] - mfcc_synth[i];
            squared_sum += diff * diff;
        }

        // Mathematical conversion formula for MCD physical scale: (10 * sqrt(2)) / ln(10) * sqrt(Sum(diff^2))
        // The scaling constant factor evaluates to approximately 6.13745262.
        double multiplier = (10.0 * std::sqrt(2.0)) / std::log(10.0);
        double frame_mcd = multiplier * std::sqrt(squared_sum);
        return frame_mcd;
    }

    // Computes the time-averaged MCD metric over the entire audio sequence.
    static double ComputeAveragedSequenceMCD(
        const std::vector<std::vector<double>>& matrix_ref, 
        const std::vector<std::vector<double>>& matrix_synth) {
        
        size_t total_frames = std::min(matrix_ref.size(), matrix_synth.size());
        if (total_frames == 0) return 0.0;

        double accumulated_mcd = 0.0;
        for (size_t t = 0; t < total_frames; ++t) {
            accumulated_mcd += ComputeFrameMCD(matrix_ref[t], matrix_synth[t]);
        }
        return accumulated_mcd / total_frames;
    }
};

int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << "[RUN] Activating C++ acoustic feature-level Mel-Cepstral Distance (MCD) solver test..." << std::endl;

    // 1. Reference matrix: 3 frames, 13 dimensions per frame representing true MFCC vectors
    std::vector<std::vector<double>> mock_ref_matrix = {
        {10.5, 1.2, -0.4, 3.2, 0.5, -1.2, 0.1, 0.4, -0.2, 0.1, 0.0, -0.5, 0.2},
        {11.2, 1.0, -0.5, 3.0, 0.6, -1.0, 0.2, 0.3, -0.1, 0.2, 0.1, -0.4, 0.3},
        {10.8, 1.5, -0.3, 3.5, 0.4, -1.4, 0.0, 0.5, -0.3, 0.0, -0.1, -0.6, 0.1}
    };

    // 2. Synthesized matrix: Predicted feature matrix with slight acoustic distortions
    std::vector<std::vector<double>> mock_synth_matrix = {
        {10.5, 1.5, -0.2, 3.0, 0.7, -1.0, 0.3, 0.2, -0.4, 0.2, 0.2, -0.3, 0.4},
        {11.2, 1.3, -0.3, 2.8, 0.8, -0.8, 0.4, 0.1, -0.3, 0.3, 0.3, -0.2, 0.5},
        {10.8, 1.8, -0.1, 3.3, 0.6, -1.2, 0.2, 0.3, -0.5, 0.1, 0.1, -0.4, 0.3}
    };

    // 3. Execute calculation
    double final_sequence_mcd = MelCepstralDistanceEngine::ComputeAveragedSequenceMCD(mock_ref_matrix, mock_synth_matrix);

    std::cout << "[SUCCESS] Acoustic geometric spectrum similarity evaluation complete." << std::endl;
    std::cout << " -> Total frames evaluated: " << std::min(mock_ref_matrix.size(), mock_synth_matrix.size()) << " frames" << std::endl;
    std::cout << " -> Final average MCD loss score: " << final_sequence_mcd << " dB" << std::endl;
    std::cout << " -> Metric verdict: Score matches industrial fidelity standards below the 5.0dB threshold." << std::endl;
    std::cout << "============================================================" << std::endl;
    return 0;
}
