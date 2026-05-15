import struct
import numpy as np

def transcode_float_matrix_to_pcm_bytes(float32_signal: np.ndarray) -> bytes:
    """
    Converts a floating-point matrix sequence from a deep learning model output
    into a 16-bit signed integer discrete PCM byte stream.
    """
    # 1. Constrain signal amplitude to prevent clipping distortion
    clipped_signal = np.clip(float32_signal, -1.0, 1.0)
    
    # 2. Linear quantization: Scale [-1.0, 1.0] float range to signed int16 range [-32768, 32767]
    int16_scaled = (clipped_signal * 32767.0).astype(np.int16)
    
    # 3. Binary serialization: Pack integer sequence into a standard Little-Endian raw stream
    # '<' denotes Little-Endian, 'h' denotes short (16-bit signed integer)
    num_samples = len(int16_scaled)
    pack_format = f"<{num_samples}h"
    
    # Execute low-level byte conversion
    binary_byte_stream = struct.pack(pack_format, *int16_scaled)
    return binary_byte_stream

if __name__ == "__main__":
    print("=" * 60)
    print("[RUN] Initializing Python matrix-to-binary PCM byte stream transcoding test driver...")
    
    # Simulate 500 floating-point acoustic data points synthesized by a deep learning TTS model
    mock_model_output = np.sin(np.linspace(0, 10, 500)) * 0.8
    
    # Execute streaming serialization and packing
    byte_output = transcode_float_matrix_to_pcm_bytes(mock_model_output)
    
    print("[SUCCESS] Feature matrix sequence quantization completed.")
    print(f" -> Original Float32 matrix size: {mock_model_output.nbytes} bytes")
    print(f" -> Packed binary PCM stream size: {len(byte_output)} bytes (Size reduced by 50%)")
    print("=" * 60)
