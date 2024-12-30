import torch
from torch.utils.data import DataLoader
from dataset import CroppedImageDataset, data_transform
from FocusNet import FocusNet
from utils import display_resized_image_with_bbox

if __name__ == "__main__":
    # Paths
    image_folder = "dataset_cropped/"
    annotations_file = "dataset_cropped/instances_train.json"

    # Load dataset
    train_dataset = CroppedImageDataset(image_folder, annotations_file, transform=data_transform)

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Load model
    model = FocusNet()

    # Test pipeline
    for images, bboxes in train_loader:
        print("Image batch shape:", images.shape)  # Should be [32, 3, 124, 124]
        print("BBox batch shape:", bboxes.shape)  # Should be [32, 4]
        break

    # Test display
    display_resized_image_with_bbox(train_dataset, idx=10)
