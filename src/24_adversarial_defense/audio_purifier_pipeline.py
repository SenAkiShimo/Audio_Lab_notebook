import torch

class AudioPurificationPipeline:
    def __init__(self, bit_rate_reduction=8):
        """
        Initializes the audio defense pipeline.
        bit_rate_reduction: Target bit-depth to compress continuous float amplitudes into 
                            coarse-grained steps to remove adversarial perturbations.
        """
        self.target_bits = bit_rate_reduction
        print(f"[GUARD_INIT] Audio purification pipeline deployed | Target bit-depth: {self.target_bits}")

    def purify_incoming_stream(self, incoming_audio_tensor):
        """
        Purifies the incoming audio stream by applying random dithering and bit-depth squeezing 
        to erase fine-grained adversarial structures.
        
        incoming_audio_tensor shape: [Batch=1, Samples=16000] with values in range [-1.0, 1.0] (Float32).
        """
        # --- Phase 1: Random Dithering ---
        # Add a low-level Gaussian white noise to disrupt the precise alignment of adversarial perturbations.
        noise_level = 0.001
        dither_noise = torch.randn_like(incoming_audio_tensor) * noise_level
        step1_audio = incoming_audio_tensor + dither_noise

        # --- Phase 2: Bit-depth Squeezing ---
        # 1. Map the continuous Float32 range [-1.0, 1.0] linearly into target discrete intervals.
        levels = 2 ** (self.target_bits - 1)  # 8-bit corresponds to 128 positive/negative quantization levels
        quantized_audio = torch.round(step1_audio * levels)

        # 2. Map back to the standard Float32 space. 
        # This round-trip quantization effectively eliminates low-amplitude adversarial trails.
        purified_audio = quantized_audio / levels

        # 3. Clip the values to maintain the valid physical amplitude range [-1.0, 1.0]
        purified_audio = torch.clamp(purified_audio, -1.0, 1.0)
        
        print("[PURIFY ENGINE] Audio stream processing completed.")
        return purified_audio

# Test driver configuration
if __name__ == "__main__":
    print("=" * 60)
    print("[RUN] Testing PyTorch Input Stream Purification Pipeline...")
    
    pipeline_guard = AudioPurificationPipeline(bit_rate_reduction=8)

    # Generate a mock poisoned audio tensor containing an adversarial bias
    torch.manual_seed(123)
    mock_poisoned_audio = torch.randn(1, 16000) * 0.1 + 0.005

    # Pass the audio through the purification pipeline before feeding it to the neural network
    safe_clean_audio = pipeline_guard.purify_incoming_stream(mock_poisoned_audio)

    # Verify the purification effects
    diff_noise = torch.abs(mock_poisoned_audio - safe_clean_audio)
    
    print("[SUCCESS] Audio purification completed successfully.")
    print(f" -> Average residual energy removed by the pipeline: {torch.mean(diff_noise).item():.6f}")
    print("=" * 60)
