import torch
import torch.nn as nn

class AudioKeywordsClassifier(nn.Module):
    """
    A simulated 3-class Keyword Spotting (KWS) neural network.
    Input: [Batch=1, 16000 time-domain samples] (1-second audio)
    Output: [Batch=1, 3 class logits] (0=white noise, 1=standard keyword, 2=malicious command)
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(16000, 128),
            nn.Tanh()
        )
        self.head = nn.Linear(128, 3)

    def forward(self, x):
        return self.head(self.encoder(x))

def generate_targeted_audio_fgsm(clean_audio: torch.Tensor, target_class=2, epsilon=0.003):
    """
    Generates an adversarial audio sample using the Targeted FGSM algorithm by 
    modifying the input waveform along the negative gradient direction of the target loss.
    """
    # 1. Instantiate the victim model and freeze all weights for white-box attack simulation
    victim_model = AudioKeywordsClassifier()
    victim_model.eval()
    for param in victim_model.parameters():
        param.requires_grad = False

    # 2. Clone the input tensor and enable gradient tracking on the waveform itself
    perturbed_audio = clean_audio.clone().detach().requires_grad_(True)

    # 3. Forward pass to compute logits
    logits = victim_model(perturbed_audio)

    # 4. Construct cross-entropy loss against the target class
    criterion = nn.CrossEntropyLoss()
    target_tensor = torch.tensor([target_class], dtype=torch.long)
    loss = criterion(logits, target_tensor)

    # 5. Backward pass to compute gradients with respect to the input waveform
    victim_model.zero_grad()
    loss.backward()

    # 6. Apply targeted adversarial step
    # For targeted attacks, we subtract the gradient sign to minimize the loss toward the target class
    audio_grad_signs = perturbed_audio.grad.data.sign()
    adversarial_wave = perturbed_audio.data - epsilon * audio_grad_signs

    # 7. Bound the waveform amplitude to the valid physical range [-1.0, 1.0]
    adversarial_wave = torch.clamp(adversarial_wave, -1.0, 1.0)

    # Verify the attack success rate by checking post-attack confidence
    post_logits = victim_model(adversarial_wave)
    success_rate = torch.softmax(post_logits, dim=-1)[0, target_class].item()
    
    print(f"[ATTACK ENGINE] FGSM generation completed | Step size epsilon = {epsilon}")
    print(f" -> Target class index: {target_class} | Target class confidence: {success_rate * 100:.2f}%")
    
    return adversarial_wave

# Test driver configuration
if __name__ == "__main__":
    print("=" * 60)
    print("[RUN] Testing Time-Domain Targeted FGSM Adversarial Audio Generator...")
    
    # Simulate 1 second of clean mono audio waveform tensor
    torch.manual_seed(42)
    mock_clean_pcm = torch.randn(1, 16000) * 0.1
    
    # Generate adversarial audio sample targeting class 2
    adv_audio = generate_targeted_audio_fgsm(mock_clean_pcm, target_class=2, epsilon=0.003)
    print("=" * 60)
