import torch
import torch.nn as nn

class FocusNet(nn.Module):
    def __init__(self):
        super(FocusNet, self).__init__()
        # Encoder: Feature extraction
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # 124x124 -> 124x124
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 124x124 -> 62x62
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 62x62 -> 31x31
            nn.ReLU()
        )
        
        # Decoder: Spatial context preservation via upsampling
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 31x31 -> 31x31
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),  # 31x31 -> 31x31
            nn.ReLU(),
        )
        
        # Fully connected head for bounding box regression
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 31 * 31, 256),
            nn.ReLU(),
            nn.Dropout(0.5),  # Regularization to prevent overfitting
            nn.Linear(256, 4)  # Output 4 values: [x, y, w, h]
        )
        
    def forward(self, x):
        # Feature extraction
        x = self.encoder(x)
        # Context refinement
        x = self.decoder(x)
        # Bounding box regression
        x = self.fc(x)
        return x