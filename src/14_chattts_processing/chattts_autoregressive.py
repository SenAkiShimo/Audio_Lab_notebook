import torch
import torch.nn as nn


class MockChatTTSTransformerDecoder(nn.Module):
    """
    Simulates the core probability space of a ChatTTS decoder-only autoregressive network.
    Maps text sequences and discrete audio neural codec tokens into a unified vocabulary
    context window to perform causal next-token prediction.
    """

    def __init__(self, joint_vocab_size=2000, hidden_dim=256):
        super().__init__()
        # Unified cross-modal embedding layer.
        # First 1000 IDs map to text tokens, remaining 1000 IDs map to discrete audio codec tokens.
        self.joint_embeddings = nn.Embedding(joint_vocab_size, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, joint_vocab_size)

    def forward(self, input_ids, paralinguistic_bias_tensor):
        """
        input_ids: Unified context sequence tensor, shape [Batch=1, Current_Sequence_Length]
        paralinguistic_bias_tensor: Probability space bias for paralinguistic event tokens (e.g., laughter, pause)
        """
        # 1. Extract cross-modal joint features
        embeddings = self.joint_embeddings(input_ids)

        # 2. Extract the hidden state of the last token for causal autoregressive deduction
        last_token_hidden = embeddings[:, -1, :]

        # 3. Project to log probabilities covering the joint space
        logits = self.lm_head(last_token_hidden)

        # 4. Multimodal flow control: Logits manipulation
        # Add paralinguistic event biases to shift the acoustic probability distribution
        manipulated_logits = logits + paralinguistic_bias_tensor
        return manipulated_logits


class ChatTTSEventFlowManager:

    def __init__(self):
        # Joint vocabulary special segment
        self.special_tokens = {
            "[laughter]": 991,  # Event tag for laughter
            "[uv_break]": 992,  # Event tag for unvoiced breath break
            "<|SOA|>": 1000,  # Start of Audio
            "<|EOA|>": 1999,  # End of Audio
        }
        self.decoder = MockChatTTSTransformerDecoder(joint_vocab_size=2000)
        print(
            "[INIT] ChatTTS causal core initialized. Joint vocabulary mapping setup complete."
        )

    def generate_paralinguistic_speech(self, text_token_sequence, inject_event=None):
        """
        Simulates paralinguistic-conditioned autoregressive generation.
        text_token_sequence: List of discrete integer IDs representing the text tokens
        inject_event: Paralinguistic event to inject ("[laughter]" or "[uv_break]")
        """
        # 1. Simulate text refinement operator
        extended_sequence = list(text_token_sequence)
        if inject_event in self.special_tokens:
            # Insert the event tag as a conditioning prefix token at the end of the text sequence
            extended_sequence.append(self.special_tokens[inject_event])

        # 2. Append the Start of Audio token to merge the text and acoustic spaces
        extended_sequence.append(self.special_tokens["<|SOA|>"])

        # Convert to tensor to initialize autoregressive context
        current_context = torch.tensor([extended_sequence], dtype=torch.long)

        # 3. Construct the paralinguistic event conditioning bias tensor
        paralinguistic_bias = torch.zeros(1, 2000)
        if inject_event == "[laughter]":
            # Boost probabilities for codec tokens corresponding to high-order harmonics and vibrato
            paralinguistic_bias[0, 1200:1400] = 20.0
            print(
                "[EVENT_INJECT] Intercepted [laughter] command. Applied 200-dim adaptive gain bias."
            )
        elif inject_event == "[uv_break]":
            # Prioritize zero-energy and silent tokens to suppress semantic pronunciation
            paralinguistic_bias[0, 1001:1050] = 40.0
            print(
                "[EVENT_INJECT] Intercepted [uv_break] command. Suppressed semantic energy for breath pause."
            )

        generated_audio_codecs = []

        # 4. Causal autoregressive next-token prediction loop
        for step in range(6):
            with torch.no_grad():
                logits = self.decoder(current_context, paralinguistic_bias)

            # Perform greedy sampling to extract the target acoustic token
            next_codec_token = torch.argmax(logits, dim=-1).item()
            generated_audio_codecs.append(next_codec_token)

            # Append the predicted audio token to the sequence to update the forward context
            next_token_tensor = torch.tensor(
                [[next_codec_token]], dtype=torch.long
            )
            current_context = torch.cat(
                [current_context, next_token_tensor], dim=1
            )

        print(
            f"[SUCCESS] Autoregressive loop complete. Final context window shape: {list(current_context.shape)}"
        )
        print(f" -> Generated discrete codec sequence: {generated_audio_codecs}")
        return generated_audio_codecs


if __name__ == "__main__":
    print("=" * 60)
    print("[RUN] Activating ChatTTS autoregressive sequence testing...")
    manager = ChatTTSEventFlowManager()

    # Mock token sequence representing input text tokens
    mock_text_ids = [12, 45, 88]

    # Scenario 1: Inject laughter token to guide expression at the end of speech
    manager.generate_paralinguistic_speech(
        mock_text_ids, inject_event="[laughter]"
    )

    # Scenario 2: Inject unvoiced break token to trigger a physiological breathing pause
    manager.generate_paralinguistic_speech(
        mock_text_ids, inject_event="[uv_break]"
    )
    print("=" * 60)
