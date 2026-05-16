import torch
import torch.nn as nn

class DeepfakeArtifactDetectorNet(nn.Module):
    """
    A 2D-CNN classification network for identifying deepfake audio synthetic artifacts.
    Input: [Batch=1, Channels=1, Freq_Bins=80, Time_Frames=300] (Acoustic spectrogram)
    Output: [Batch=1, 1] (Probability of being a deepfake spoof, ranging from 0.0 to 1.0)
    """
    def __init__(self):
        super().__init__()
        
        # 2D Convolutional layers to scan for chess-board patterns and structural anomalies
        self.grating_scanner = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Squeezes spatial dimensions to handle variable-length inputs
        )
        
        self.decision_head = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, spectrogram_tensor):
        """
        Standard forward pass for spoof detection.
        """
        # 1. Extract high-frequency features via convolution layers
        extracted_features = self.grating_scanner(spectrogram_tensor)
        
        # 2. Flatten tensor dimensions from [Batch, 32, 1, 1] to [Batch, 32]
        flattened_features = torch.flatten(extracted_features, start_dim=1)
        
        # 3. Compute binary decision logits and squash to a probability value
        logits = self.decision_head(flattened_features)
        deepfake_score = self.sigmoid(logits)
        
        return deepfake_score

# Test driver configuration
if __name__ == "__main__":
    print("=" * 60)
    print("[RUN] Testing Python 2D-CNN Deepfake Audio Artifact Detector...")
    
    anti_fraud_guard = DeepfakeArtifactDetectorNet()
    anti_fraud_guard.eval()

    # Simulate 1 frame of an 80-bin Mel-spectrogram with 300 time frames
    torch.manual_seed(777)
    mock_incoming_spec = torch.randn(1, 1, 80, 300) * 0.5

    # Execute deepfake verification inference
    with torch.no_grad():
        forgery_probability = anti_fraud_guard(mock_incoming_spec)
        prob_val = forgery_probability.item()

    print("[SUCCESS] Spectrogram scanning completed.")
    print(f" -> Input spectrogram shape: {list(mock_incoming_spec.shape)}")
    print(f" -> Deepfake Score: {prob_val * 100:.2f}%")
    print(f" -> Verdict: {'REJECT (Deepfake Audio Detected)' if prob_val >= 0.5 else 'PASS (Natural Audio Confirmed)'}")
    print("=" * 60)
