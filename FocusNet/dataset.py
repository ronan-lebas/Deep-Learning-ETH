import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
from torchvision import transforms

# Data Transform
data_transform = transforms.Compose([
    transforms.Resize((124, 124)),  # Resize to 124x124
    transforms.ToTensor()
])

class CroppedImageDataset(Dataset):
    def __init__(self, image_folder, annotations_file, transform=data_transform):
        self.image_folder = image_folder
        self.annotations = json.load(open(annotations_file))["annotations"]
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_path = os.path.join(self.image_folder, annotation["file_name"])
        image = Image.open(img_path).convert("RGB")

        # Original image dimensions
        original_width, original_height = image.size

        # Original bbox in [xmin, ymin, w, h]
        xmin, ymin, box_width, box_height = annotation["bbox"]

        # Convert bbox to YOLO format: [center_x, center_y, w, h] and normalize
        center_x = (xmin + box_width / 2) / original_width
        center_y = (ymin + box_height / 2) / original_height
        norm_width = min(box_width / original_width, 1)
        norm_height = min(box_height / original_height, 1)

        bbox = torch.tensor([center_x, center_y, norm_width, norm_height], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, bbox