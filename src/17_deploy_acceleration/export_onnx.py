import os
import torch
import torch.nn as nn


class AudioEncoderModule(nn.Module):
    """
    Simulates a 1D convolutional acoustic feature extraction frontend for deep learning speech models.
    """

    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(
                in_channels=80,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GELU(),
        )
        self.fc_head = nn.Linear(256, 128)

    def forward(self, x):
        # Expected input shape: [Batch, 80_Mel_Channels, Variable_Time_Frames]
        x = self.conv_block(x)
        x = x.transpose(1, 2)  # Transpose to match linear layer dimensions
        return self.fc_head(x)


if __name__ == "__main__":
    print("=" * 60)
    print(
        "[RUN] Activating Python dynamic time axes binding and ONNX static graph export test..."
    )

    # 1. Initialize acoustic model and switch to evaluation mode
    model = AudioEncoderModule()
    model.eval()

    # 2. Construct a dummy input tensor for tracing
    # Shape: Batch size 1, 80 mel channels, 200 time frames
    dummy_audio_input = torch.randn(1, 80, 200)

    # 3. Define dynamic axes configuration to ensure flexible deployment shapes
    dynamic_axes_spec = {
        "input_mel_spectrogram": {0: "batch_size", 2: "dynamic_time_frames"},
        "output_latent_features": {1: "dynamic_time_frames", 0: "batch_size"},
    }

    # 4. Set target output filename and ensure directory structure exists
    output_filename = "src/17_deploy_acceleration/audio_encoder.onnx"
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 5. Export static Directed Acyclic Graph (DAG) to ONNX format
    torch.onnx.export(
        model,
        dummy_audio_input,
        output_filename,
        export_params=True,  # Serialize model parameters directly inside the binary graph
        opset_version=17,  # Use a modern opset version supporting complex sequence operators
        input_names=["input_mel_spectrogram"],
        output_names=["output_latent_features"],
        dynamic_axes=dynamic_axes_spec,
    )

    print(
        f"[SUCCESS] ONNX graph exported successfully with dynamic variables. Saved to {output_filename}"
    )
    print("=" * 60)
