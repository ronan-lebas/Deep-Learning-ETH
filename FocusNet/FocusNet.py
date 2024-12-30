import torch
import torch.nn as nn

class FocusNet(nn.Module):
    def __init__(self):
        super(FocusNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 124 -> 62
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 62 -> 31
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 31 * 31, 256),
            nn.ReLU(),
            nn.Linear(256, 4)  # Output 4 values: [x, y, w, h]
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x