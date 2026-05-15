import torch
import torch.nn as nn

class MockWhisperDecoderCrossAttention(nn.Module):
    """
    Simulates the autoregressive multi-class logit computation kernel of a Whisper decoder.
    Injects specific special tokens at the sequence prefix to guide cross-modal attention bias.
    """
    def __init__(self, vocab_size=50, embedding_dim=128):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Linear projection layer simulating cross-attention alignment between acoustic vectors and text tokens
        self.cross_attention_projector = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_token_ids, task_bias_tensor):
        """
        input_token_ids: Generated token ID sequence of shape [Batch=1, Current_Sequence_Length]
        task_bias_tensor: Task-specific conditioning bias matrix of shape [1, Vocab_Size]
        """
        # 1. Extract token hidden embedding vectors
        x_emb = self.token_embeddings(input_token_ids)
        
        # 2. Slice the hidden state of the final frame to perform next-token autoregressive prediction
        last_frame_hidden = x_emb[:, -1, :]
        logits = self.cross_attention_projector(last_frame_hidden)
        
        # 3. Apply task-specific conditioning constraint matrix bias to manipulate probability distribution
        manipulated_logits = logits + task_bias_tensor
        return manipulated_logits

class WhisperSpecialTokenFlowManager:
    def __init__(self):
        # Explicit special tokens dictionary mapping table
        self.special_tokens = {
            "<|startoftranscript|>": 1,
            "<|zh|>": 2,
            "<|en|>": 3,
            "<|transcribe|>": 4,
            "<|translate|>": 5,
            "<|endoftext|>": 0
        }
        self.decoder_core = MockWhisperDecoderCrossAttention(vocab_size=50)
        self.decoder_core.eval() # Set model to evaluation mode for inference stability
        print("[INIT] Whisper multitask flow control engine initialized. Special token mapping aligned.")

    def simulate_multitask_inference(self, target_language="<|zh|>", task_mode="<|transcribe|>"):
        """
        Executes a simulated autoregressive generation sequence guided by special token prefix prompts.
        """
        print(f"\n[TASK STAGE] Initializing decode stream -> Target Language: {target_language} | Task Mode: {task_mode}")
        
        # 1. Construct the sequence prefix prompt: [SOT] -> [Language] -> [Task]
        prompt_prefix = [
            self.special_tokens["<|startoftranscript|>"],
            self.special_tokens[target_language],
            self.special_tokens[task_mode]
        ]
        current_sequence = torch.tensor([prompt_prefix], dtype=torch.long)
        
        # 2. Configure mock cross-modal attention conditioning bias matrices based on the target task
        task_bias = torch.zeros(1, 50)
        if task_mode == "<|transcribe|>":
            task_bias[0, 10:20] = 15.0  # Elevate activation space for original source tokens
        elif task_mode == "<|translate|>":
            task_bias[0, 30:40] = 15.0  # Elevate activation space for target translated tokens
            
        generated_tokens = []
        
        # 3. Autoregressive causal generation loop
        for step in range(5):
            with torch.no_grad():
                logits = self.decoder_core(current_sequence, task_bias)
            
            # Execute greedy sampling to extract the target token index
            next_token_id = torch.argmax(logits, dim=-1).item()
            generated_tokens.append(next_token_id)
            
            # FIX 1 & 2: Properly wrap the scalar ID into a 2D batch tensor and concatenate to update the context window
            next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long)
            current_sequence = torch.cat([current_sequence, next_token_tensor], dim=1)
            
        print(f"[SUCCESS] Inference complete. Prefix: {prompt_prefix} -> Decoded Tokens: {generated_tokens}")
        return generated_tokens

if __name__ == "__main__":
    print("=" * 60)
    print("[RUN] Activating Python special token autoregressive control flow test driver...")
    
    manager = WhisperSpecialTokenFlowManager()
    
    # Evaluation Scenario A: Transcription task targeting native source language extraction
    manager.simulate_multitask_inference(target_language="<|zh|>", task_mode="<|transcribe|>")
    
    # Evaluation Scenario B: Translation task targeting cross-modal English output alignment
    manager.simulate_multitask_inference(target_language="<|zh|>", task_mode="<|translate|>")
    
    print("=" * 60)
