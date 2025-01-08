import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class FocusNetv3(nn.Module):
    def __init__(self):
        super(FocusNetv3, self).__init__()

        # Pretrained ResNet Backbone
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.encoder1 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu
        )  # 64 channels
        self.encoder2 = nn.Sequential(resnet.maxpool, resnet.layer1)  # 64 channels
        self.encoder3 = resnet.layer2  # 128 channels
        self.encoder4 = resnet.layer3  # 256 channels

        # Decoder with Skip Connections
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decode3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decode2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decode1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.channel_reduce = nn.Conv2d(96, 64, kernel_size=1)

        # Fully Connected Layers for Bounding Box Regression
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((16, 16)),  # Reduce spatial dimensions to 16x16 (was 64x64, but computational instabilities)
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 512),  # Adjust input size
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 4),  # Output: [x, y, w, h]
        )

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        # Decoder
        up3 = self.upconv3(enc4)
        dec3 = self.decode3(torch.cat([up3, enc3], dim=1))

        up2 = self.upconv2(dec3)
        enc2 = F.pad(enc2, (0, 1, 0, 1))
        dec2 = self.decode2(torch.cat([up2, enc2], dim=1))

        up1 = self.upconv1(dec2)
        enc1 = F.pad(enc1, (1, 1, 1, 1))
        x = torch.cat([up1, enc1], dim=1)
        x = self.channel_reduce(x)
        dec1 = self.decode1(x)

        # Bounding Box Regression
        x = self.fc(dec1)
        return x
