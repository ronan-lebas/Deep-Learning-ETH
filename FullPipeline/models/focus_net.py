import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms


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
            resnet.layer3,  # 256 channels
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
            nn.ReLU(),
        )

        # Fully Connected Layers for Bounding Box Regression
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 4),  # Output: [x, y, w, h]
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

    def label_to_bbox(
        self, label: torch.Tensor, img_width: int, img_height: int
    ) -> torch.Tensor:
        """
        Convert YOLO label format to bounding box coordinates.

        Args:
            label (torch.Tensor): Label in YOLO format (center_x, center_y, width, height),
                                normalized by the image dimensions.
            img_width (int): Width of the image.
            img_height (int): Height of the image.

        Returns:
            torch.Tensor: Bounding box in (x, y, w, h) format where x, y are the top-left coordinates.
        """
        center_x, center_y, w, h = label.unbind(0)
        center_x = center_x.clone()
        center_y = center_y.clone()
        w = w.clone()
        h = h.clone()

        # Denormalize the coordinates
        center_x *= img_width
        center_y *= img_height
        w *= img_width
        h *= img_height

        # Calculate the top-left corner
        x = center_x - w / 2.0
        y = center_y - h / 2.0

        x = torch.clamp(x, 0.0, img_width)
        y = torch.clamp(y, 0.0, img_height)
        w = torch.clamp(w, 0.0, img_width)
        h = torch.clamp(h, 0.0, img_height)

        return torch.tensor([x, y, w, h])

    def refine_bounding_box(self, image, bbox):
        """
        Refines the bounding box using the FocusNet model.

        Args:
            image: a PIL Image
            bbox: initial bounding box in (left, top, width, height) format.

        Returns:
            Refined bounding box in (left, top, width, height) format.
        """
        # Extract initial bounding box
        left, top, width, height = bbox
        cropped_image = image.crop((left, top, left + width, top + height)).convert(
            "RGB"
        )

        # Transformations: Convert to tensor and resize to 124x124
        transform = transforms.Compose(
            [
                transforms.Resize((124, 124)),
                transforms.ToTensor(),
            ]
        )
        input_tensor = transform(cropped_image).unsqueeze(0)  # Add batch dimension

        # Process the image using the model
        output_label = self(input_tensor)

        # Denormalize and convert to bounding box format
        refined_bbox = self.label_to_bbox(output_label.squeeze(), 124, 124)

        # Adjust coordinates back to the original scale
        resize_ratio_width = width / 124.0
        resize_ratio_height = height / 124.0

        refined_left = refined_bbox[0].item() * resize_ratio_width + left
        refined_top = refined_bbox[1].item() * resize_ratio_height + top
        refined_width = refined_bbox[2].item() * resize_ratio_width
        refined_height = refined_bbox[3].item() * resize_ratio_height

        return (refined_left, refined_top, refined_width, refined_height)
