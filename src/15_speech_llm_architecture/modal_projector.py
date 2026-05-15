import torch
import torch.nn as nn


class AudioTextModalProjector(nn.Module):
    """
    Bridge core for streaming multimodal speech LLMs: Modal Adapter.
    Projects 1024-dimensional dense feature vectors from the acoustic encoder
    into the 4096-dimensional geometric space of the LLM text tokens.
    """

    def __init__(self, audio_dim=1024, text_dim=4096):
        super().__init__()
        # Two-layer feed-forward network (Projector MLP) with activation function.
        self.projector = nn.Sequential(
            nn.Linear(audio_dim, text_dim),
            nn.GELU(),  # Gaussian Error Linear Unit for non-linear transformation
            nn.Linear(text_dim, text_dim),
        )
        print(
            f"[PROJECTOR_INIT] Cross-modal transformation matrix ready | {audio_dim}D -> {text_dim}D"
        )

    def forward_alignment(self, raw_audio_features):
        """
        raw_audio_features: Original continuous matrix from the acoustic encoder,
                            shape [Batch=1, Audio_Frames=150, Audio_Dim=1024]
        """
        # 1. Perform matrix projection to align with the high-dimensional text space.
        # Shape transforms from [1, 150, 1024] to [1, 150, 4096].
        aligned_audio_embeddings = self.projector(raw_audio_features)

        # 2. Verify dimension consistency
        assert (
            aligned_audio_embeddings.shape[-1] == 4096
        ), "Modal mapping anomaly: Dimension does not match the LLM embedding size."

        print(
            f"[MODAL_ADAPTER] Spatial alignment successful | Feature matrix shape: {list(aligned_audio_embeddings.shape)}"
        )
        return aligned_audio_embeddings


if __name__ == "__main__":
    print("=" * 60)
    print("[RUN] Activating cross-modal feature alignment projector testing...")

    adapter = AudioTextModalProjector(audio_dim=1024, text_dim=4096)
    adapter.eval()

    # Simulate 1.5 seconds of audio: 150 frames, 1024 dimensions per frame.
    mock_raw_audio = torch.randn(1, 150, 1024) * 0.5

    # Execute modal mapping
    with torch.no_grad():
        aligned_matrix = adapter.forward_alignment(mock_raw_audio)

    print(
        "[SUCCESS] Audio features projected to 4096D space, ready for the autoregressive network."
    )
    print("=" * 60)
