import os
import json
import numpy as np
from tqdm import tqdm
from shutil import copyfile
from PIL import Image

# Function to convert bounding box to YOLO format
def bbox_to_label(bbox, img_width, img_height):
    """
    Convert bounding box coordinates to YOLO label format.

    Args:
        bbox (list): Bounding box in (x, y, w, h) format where x, y are the top-left coordinates.
        img_width (int): Width of the image.
        img_height (int): Height of the image.

    Returns:
        list: Bounding box in YOLO format (center_x, center_y, width, height), normalized by the image dimensions.
    """
    x, y, w, h = bbox

    # Compute center coordinates
    x_center = x + w / 2.0
    y_center = y + h / 2.0

    # Normalize coordinates
    x_center /= img_width
    y_center /= img_height
    w /= img_width
    h /= img_height

    return np.clip([x_center, y_center, w, h], 0.0, 1.0)

# Paths
original_dataset = "augmented_dataset"  # Replace with your original dataset folder
input_file = os.path.join(original_dataset, "augmented_trashcan.json")
output_dir = "augmented_dataset_YOLO"  # Replace with your output directory

train_images_dir = os.path.join(output_dir, "train/images")
train_labels_dir = os.path.join(output_dir, "train/labels")

# Create directories
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)

# Load the dataset
with open(input_file, "r") as f:
    data = json.load(f)

# Split the data (80% train, 20% val)
data_len = len(data)

# Process data
def process_data(data, images_dir, labels_dir):
    for item in tqdm(data, desc=f"Processing {images_dir}"):
        # Read data
        image_id = item["image_id"]
        original_bbox = item["refined_bbox"]  # Use refined_bbox for YOLO
        category = item["category"] - 1 # YOLO format starts from 0
        filename = item["filename"]
        
        # open the image file with pil and get its dimensions
        img = Image.open(os.path.join(original_dataset, filename))
        
        # Assuming image dimensions are known (replace with actual values if available)
        img_width, img_height = img.size

        # Convert bbox to YOLO format
        x, y, w, h = bbox_to_label(original_bbox, img_width, img_height)

        # Write label file
        label_file = os.path.join(labels_dir, f"{os.path.splitext(filename)[0]}.txt")
        with open(label_file, "w") as lf:
            lf.write(f"{category} {x} {y} {w} {h}\n")

        # Copy image file
        src_image_path = os.path.join(original_dataset, filename)
        dst_image_path = os.path.join(images_dir, filename)
        if os.path.exists(src_image_path):
            copyfile(src_image_path, dst_image_path)
        else:
            print(f"Warning: {src_image_path} does not exist!")

# Process train and val datasets
process_data(data, train_images_dir, train_labels_dir)

print("Dataset conversion completed!")
