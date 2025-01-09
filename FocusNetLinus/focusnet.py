import json
import os
from PIL import Image
from torch.utils.data import Dataset
import random
import torch
import torch.nn as nn
from torchvision.ops import roi_align
import torchvision.models as models


class FocusNetDataset(Dataset):
    def __init__(self, annotation_file, image_dir, transform=None, expansion_ratio=0.2):
        with open(annotation_file, 'r') as f:
            data = json.load(f)

        self.images = {img['id']: img for img in data['images']}
        self.annotations = []

        # Flatten dataset: one annotation per entry
        for ann in data['annotations']:
            image_id = ann['image_id']
            if image_id in self.images:
                self.annotations.append({
                    'image_id': image_id,
                    'file_name': self.images[image_id]['file_name'],
                    'original_bbox': ann['original_bbox']
                })

        self.image_dir = image_dir
        self.transform = transform
        self.expansion_ratio = expansion_ratio

    def __len__(self):
        return len(self.annotations)

    def expand_bbox(self, bbox, img_width, img_height):
        """
        Randomly expand a bounding box by up to the specified expansion ratio.
        """
        x, y, w, h = bbox

        expand_x = random.uniform(0, self.expansion_ratio) * w
        expand_y = random.uniform(0, self.expansion_ratio) * h

        x_new = max(0, x - expand_x)
        y_new = max(0, y - expand_y)
        w_new = min(img_width - x_new, w + 2 * expand_x)
        h_new = min(img_height - y_new, h + 2 * expand_y)

        return [x_new, y_new, w_new, h_new]

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_path = os.path.join(self.image_dir, annotation['file_name'])

        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Generate expanded bounding box dynamically
        image_id = annotation['image_id']
        image_info = self.images[image_id]
        img_width = image_info['width']
        img_height = image_info['height']

        original_bbox = annotation['original_bbox']
        expanded_bbox = self.expand_bbox(original_bbox, img_width, img_height)

        # Normalize bounding boxes
        def normalize_bbox(bbox, img_width, img_height):
            x, y, w, h = bbox
            return [
                x / img_width,
                y / img_height,
                w / img_width,
                h / img_height
            ]

        normalized_expanded_bbox = normalize_bbox(expanded_bbox, img_width, img_height)
        normalized_original_bbox = normalize_bbox(original_bbox, img_width, img_height)

        # Convert to tensors
        normalized_expanded_bbox = torch.tensor(normalized_expanded_bbox, dtype=torch.float32)
        normalized_original_bbox = torch.tensor(normalized_original_bbox, dtype=torch.float32)

        return image, normalized_expanded_bbox, normalized_original_bbox


class FocusNet(nn.Module):
    def __init__(self):
        super(FocusNet, self).__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bbox_head = nn.Sequential(
            nn.Linear(256 + 4, 128),  # +4 for expanded_bbox input
            nn.ReLU(),
            nn.Linear(128, 4)  # Output: x, y, w, h
        )

    def forward(self, image, expanded_bbox):
        features = self.backbone(image)
        pooled_features = self.global_pool(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)

        # Concatenate image features with expanded bounding box
        combined = torch.cat((pooled_features, expanded_bbox), dim=1)

        predicted_bbox = self.bbox_head(combined)
        return predicted_bbox


class FocusNetROI(nn.Module):
    def __init__(self, pool_size=(7, 7)):
        super(FocusNetROI, self).__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # We'll use ROIAlign to get a fixed-size pooled feature map
        self.pool_size = pool_size

        # After we pool the region, we flatten it
        # Suppose we want to keep a similar dimension as before, adapt as needed
        flattened_size = 256 * self.pool_size[0] * self.pool_size[1]

        # A small head to regress the bounding box
        self.bbox_head = nn.Sequential(
            nn.Linear(flattened_size + 4, 256),  # +4 for expanded_bbox
            nn.ReLU(),
            nn.Linear(256, 4)
        )

    def forward(self, images, expanded_bbox):
        """
        images: (batch_size, 3, H, W)
        expanded_bbox: (batch_size, 4) in normalized coords [x, y, w, h]
        """
        # 1. Extract feature maps
        features = self.backbone(images)

        # 2. Convert normalized boxes to absolute coordinates for ROIAlign
        #    ROIAlign expects (x1, y1, x2, y2) in the coordinate space of the input
        #    to `features`, but those features are downsampled from the input image.
        #
        #    For simplicity, let's assume we have a known scale_factor = 1/4 or 1/8
        #    depending on your backbone. (E.g. your backbone stride is 16 total?)
        #    You need to figure out the exact scale relative to your input image size.

        # Example: if input images are 224x224, your final feature map might be 14x14
        # which is a stride of 16 in each dimension. Let's call that stride = 16.
        stride = 16
        batch_boxes = []
        for i in range(images.size(0)):
            # expanded_bbox[i] = (x, y, w, h) in [0,1]
            x, y, w, h = expanded_bbox[i]
            # Convert to absolute image coords
            # image_size is 224 for example. Adjust if you do resizing differently.
            image_size = 224
            x_abs = x * image_size
            y_abs = y * image_size
            w_abs = w * image_size
            h_abs = h * image_size

            x1 = x_abs
            y1 = y_abs
            x2 = x_abs + w_abs
            y2 = y_abs + h_abs

            # Convert to feature-map coords by dividing by stride
            x1_feat = x1 / stride
            y1_feat = y1 / stride
            x2_feat = x2 / stride
            y2_feat = y2 / stride

            # ROIAlign expects bounding boxes with shape (batch_index, x1, y1, x2, y2)
            batch_boxes.append([i, x1_feat, y1_feat, x2_feat, y2_feat])

        batch_boxes = torch.tensor(batch_boxes, dtype=torch.float, device=images.device)

        # 3. ROIAlign on the feature map
        #    Output shape: (batch_size, 256, pool_size[0], pool_size[1])
        roi_features = roi_align(
            features,
            batch_boxes,
            output_size=self.pool_size,
            spatial_scale=1.0,  # Adjust if your feature map is smaller by a factor
            sampling_ratio=-1
        )

        # 4. Flatten the ROI-pooled region
        roi_features = roi_features.view(roi_features.size(0), -1)  # (batch_size, 256*poolH*poolW)

        # 5. Concatenate with the expanded bounding box
        combined = torch.cat((roi_features, expanded_bbox), dim=1)

        # 6. Regress final bounding box
        predicted_bbox = self.bbox_head(combined)

        return predicted_bbox



class FocusNetResNet50(nn.Module):
    def __init__(self, freeze_backbone=False):
        super(FocusNetResNet50, self).__init__()

        # 1. Load a pretrained ResNet50
        #    The weights argument might differ depending on your PyTorch version
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # 2. Remove the final classification layer (fc)
        #    We only need the feature-extraction part.
        #    ResNet layers up to avgpool produce (batch_size, 2048, 1, 1) after pooling
        #    or (batch_size, 2048, H, W) before the avgpool, depending on your approach.
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        # Explanation:
        # children():
        #   [Conv1, BN1, ReLU, MaxPool, Layer1, Layer2, Layer3, Layer4, AvgPool, FC]
        # We keep everything up to Layer4 (excluded the AvgPool and FC)

        # 3. Freeze some layers if desired
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 4. We'll do a global average pooling manually
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 5. Our head: We still have +4 input for bounding box coords
        #    But note the feature size from ResNet50’s Layer4 is 2048 channels, not 256.
        self.bbox_head = nn.Sequential(
            nn.Linear(2048 + 4, 512),
            nn.ReLU(),
            nn.Linear(512, 4)  # (x, y, w, h)
        )

    def forward(self, images, expanded_bbox):
        """
        images: shape (B, 3, H, W)
        expanded_bbox: shape (B, 4)
        """
        # 1. Extract feature maps from ResNet
        features = self.backbone(images)  # (B, 2048, H//32, W//32) typically

        # 2. Global average pool
        #    Now features become (B, 2048, 1, 1)
        pooled_features = self.global_pool(features)
        # Flatten to (B, 2048)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)

        # 3. Concatenate with expanded bounding box (B, 4)
        combined = torch.cat((pooled_features, expanded_bbox), dim=1)  # (B, 2048 + 4)

        # 4. Regress final bounding box
        predicted_bbox = self.bbox_head(combined)  # (B, 4)
        return predicted_bbox




class FocusNetResNet50ROI(nn.Module):
    def __init__(self, freeze_backbone=False, pool_size=(7, 7)):
        super(FocusNetResNet50ROI, self).__init__()

        # 1. Load a pretrained ResNet50
        #    Depending on your PyTorch version, you might do:
        #    resnet = models.resnet50(pretrained=True)
        #    or:
        #    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # 2. Remove the final avgpool and fc layers so we keep only the convolutional backbone
        #    children():
        #      [Conv1, BN1, ReLU, MaxPool, Layer1, Layer2, Layer3, Layer4, AvgPool, FC]
        #    We take everything up to 'layer4':
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # This is the size we want from ROIAlign
        self.pool_size = pool_size

        # If you truncated ResNet at layer4, the output channels = 2048
        # Flattened size after pooling = 2048 * poolH * poolW
        flattened_size = 2048 * pool_size[0] * pool_size[1]

        # Our bounding-box regression head
        # +4 because we’re concatenating the bounding-box coords
        self.bbox_head = nn.Sequential(
            nn.Linear(flattened_size + 4, 512),
            nn.ReLU(),
            nn.Linear(512, 4)
        )

        # We’ll store a nominal input size, e.g. 224, if your dataset is resized to 224x224.
        # If you train on different sizes, you’ll need to adjust logic in forward().
        self.input_size = 224  # or whatever your transform Resize is

        # The stride for ResNet50’s final feature map after layer4 is 32.
        # (Conv1: /2, MaxPool: /2, layer1: /4, layer2: /8, layer3: /16, layer4: /32)
        self.feature_stride = 32

    def forward(self, images, expanded_bbox):
        """
        images: (batch_size, 3, H, W)
        expanded_bbox: (batch_size, 4) in normalized coords [x, y, w, h].
        """
        # 1. Extract feature maps via ResNet (up to layer4)
        features = self.backbone(images)
        # e.g. if images are 224x224, features might be (B, 2048, 7, 7)

        # 2. Convert normalized boxes [x, y, w, h] to feature-map coords for ROIAlign
        #    ROIAlign expects bboxes in format (batch_idx, x1, y1, x2, y2) in the coordinate
        #    system of 'features'.
        batch_boxes = []
        batch_size = images.size(0)

        for i in range(batch_size):
            x, y, w, h = expanded_bbox[i]

            # Convert from normalized [0,1] to absolute input coords
            x_abs = x * self.input_size
            y_abs = y * self.input_size
            w_abs = w * self.input_size
            h_abs = h * self.input_size

            x1 = x_abs
            y1 = y_abs
            x2 = x_abs + w_abs
            y2 = y_abs + h_abs

            # Now map from input coords -> feature-map coords
            x1_feat = x1 / self.feature_stride
            y1_feat = y1 / self.feature_stride
            x2_feat = x2 / self.feature_stride
            y2_feat = y2 / self.feature_stride

            batch_boxes.append([i, x1_feat, y1_feat, x2_feat, y2_feat])

        # Convert to Tensor on the same device as images
        batch_boxes = torch.tensor(batch_boxes, dtype=torch.float, device=images.device)

        # 3. ROIAlign with output_size=self.pool_size
        #    Typically: output shape -> (total_boxes, 2048, poolH, poolW)
        #    total_boxes = batch_size if each image only has one ROI
        roi_features = roi_align(
            input=features,
            boxes=batch_boxes,
            output_size=self.pool_size,
            spatial_scale=1.0,  # we already accounted for stride manually
            sampling_ratio=-1
        )

        # 4. Flatten the ROI-pooled region
        #    shape -> (batch_size, 2048*poolH*poolW)
        roi_features = roi_features.view(roi_features.size(0), -1)

        # 5. Concatenate the bounding-box coords (in original normalized form)
        combined = torch.cat((roi_features, expanded_bbox), dim=1)

        # 6. Regress the final bounding box
        predicted_bbox = self.bbox_head(combined)

        return predicted_bbox
