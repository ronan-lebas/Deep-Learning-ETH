import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Channel-wise average
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Channel-wise max
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=(2, 3), keepdim=True)  # Spatial average
        max_out = torch.amax(x, dim=(2, 3), keepdim=True)  # Spatial max
        avg_out = self.fc2(self.relu(self.fc1(avg_out)))
        max_out = torch.amax(x, dim=(2, 3), keepdim=True)  # Spatial max
        return self.sigmoid(avg_out + max_out)

class FocusNetv4(nn.Module):
    def __init__(self):
        super(FocusNetv4, self).__init__()

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

        # Attention modules
        self.channel_attention = ChannelAttention(256)
        self.spatial_attention = SpatialAttention()

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

        # Apply Channel Attention
        enc_features = enc_features * self.channel_attention(enc_features)

        # Apply Spatial Attention
        enc_features = enc_features * self.spatial_attention(enc_features)

        # Decoder
        dec_features = self.decoder(enc_features)

        # Bounding Box Regression
        x = self.fc(dec_features)
        return x
