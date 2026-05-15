import torch
import torch.nn as nn


class HandcraftedResidualVectorQuantizer(nn.Module):
    """
    Residual Vector Quantizer (RVQ) for neural audio codecs.
    Approximates high-dimensional continuous acoustic hidden vectors into
    a sequence of discrete integer tokens through multi-stage codebook cascades.
    """

    def __init__(self, num_quantizers=4, codebook_size=1024, vector_dim=128):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.vector_dim = vector_dim

        # Construct independent, randomly initialized continuous feature codebooks.
        # Each codebook shape is [1024, 128], representing vocabulary size and feature dimension.
        self.codebooks = nn.ParameterList(
            [
                nn.Parameter(torch.randn(codebook_size, vector_dim) * 0.05)
                for _ in range(num_quantizers)
            ]
        )

    def forward_quantize(self, continuous_encoder_vector):
        """
        Forward quantization operator.
        Maps a continuous acoustic hidden vector to N discrete integer tokens
        by sequentially subtracting residuals.
        Shape of continuous_encoder_vector: [1, 128] (single frame encoder output).
        """
        discrete_tokens = []

        # Initial residual equals the input continuous hidden feature.
        current_residual = continuous_encoder_vector.clone()

        # Perform cascade quantization across multiple codebook stages.
        for q in range(self.num_quantizers):
            codebook = self.codebooks[q]

            # 1. Compute Euclidean distance (L2): ||residual - codebook_vector||^2.
            # Shape of distances: [1, 1024].
            distances = torch.cdist(current_residual, codebook, p=2)

            # 2. Nearest neighbor search to find the closest codebook vector index.
            best_match_idx = torch.argmin(distances, dim=-1).item()
            discrete_tokens.append(best_match_idx)

            # 3. Extract the selected vector and subtract it to get the next residual.
            selected_vector = codebook[best_match_idx].unsqueeze(0)
            current_residual = current_residual - selected_vector

        return discrete_tokens

    def inverse_dequantize(self, discrete_tokens_list):
        """
        Inverse dequantization operator.
        Reconstructs the continuous acoustic feature vector by looking up
        and accumulating vectors from multi-stage codebooks using the token indices.
        """
        reconstructed_vector = torch.zeros(
            1, self.vector_dim, device=self.codebooks[0].device
        )

        # Accumulate vectors: E_rebuilt = Sum(Codebook_q[Token_q]).
        for q, token_idx in enumerate(discrete_tokens_list):
            codebook = self.codebooks[q]
            selected_vector = codebook[token_idx].unsqueeze(0)
            reconstructed_vector += selected_vector

        return reconstructed_vector


if __name__ == "__main__":
    print("=" * 60)
    print("[RUN] Activating multi-stage RVQ quantization and dequantization test...")

    # Initialize RVQ: 4 stages, codebook size 1024, vector dimension 128.
    rvq_engine = HandcraftedResidualVectorQuantizer(
        num_quantizers=4, codebook_size=1024, vector_dim=128
    )
    rvq_engine.eval()

    # Simulate a single-frame hidden vector from an audio encoder.
    mock_continuous_hidden = torch.randn(1, 128) * 0.3

    # 1. Execute forward quantization (continuous to discrete).
    output_tokens = rvq_engine.forward_quantize(mock_continuous_hidden)

    # 2. Execute inverse dequantization (discrete to continuous).
    reconstructed_hidden = rvq_engine.inverse_dequantize(output_tokens)

    # 3. Calculate Mean Squared Error (MSE).
    mse_loss = torch.mean(
        (mock_continuous_hidden - reconstructed_hidden) ** 2
    ).item()

    print("[SUCCESS] RVQ bidirectional validation completed.")
    print(f" -> Original vector quantized to tokens: {output_tokens}")
    print(f" -> Reconstruction Mean Squared Error (MSE): {mse_loss:.8f}")
    print("=" * 60)
