import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class FocusNetv2(nn.Module):
    def __init__(self):
        super(FocusNetv2, self).__init__()
        
        # Pretrained ResNet Backbone
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # 64 channels
            resnet.layer2,  # 128 channels
            resnet.layer3   # 256 channels
        )
        
        # Decoder with Upsampling and Skip Connections
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # Reduce channels
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # Upsample
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # Final feature refinement
            nn.ReLU()
        )
        
        # Fully Connected Layers for Bounding Box Regression
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 4)  # Output: [x, y, w, h]
        )
    
    def forward(self, x):
        # Encoder
        enc_features = self.encoder(x)
        
        # Decoder
        dec_features = self.decoder(enc_features)
        
        # Bounding Box Regression
        x = self.fc(dec_features)
        return x