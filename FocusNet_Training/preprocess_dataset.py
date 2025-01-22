import os
import json
from PIL import Image
from tqdm import tqdm
import random

# Paths to your dataset
base_folder = "dataset/material_version/"
image_folders = {"train": os.path.join(base_folder, "train/"),
                 "val": os.path.join(base_folder, "val/")}
annotations_files = {"train": os.path.join(base_folder, "instances_train_trashcan.json"),
                     "val": os.path.join(base_folder, "instances_val_trashcan.json")}
output_folder = "dataset_cropped/"

# Parameters for minimum dimensions
# Now done directly when preprocessing the dataset
min_dim1 = 0  # Minimum for one dimension
min_dim2 = 0  # Minimum for the other dimension

# Create output directories
for split in ["train", "val"]:
    os.makedirs(os.path.join(output_folder, split), exist_ok=True)

# Function to generate adjusted crop boxes
def generate_loose_bboxes(bbox, img_width, img_height):
    x, y, w, h = bbox
    case = random.choices(
        [1, 2, 3, 4, 5],
        weights=[15, 20, 45.5, 16.25, 8.25],
        k=1
    )[0]

    if case == 1:  # 15% exact bounding box
        left, top, right, bottom = int(x), int(y), int(x + w), int(y + h)

    elif case == 2:  # 20% margin 0% to 10% in all directions
        margin_x = int(w * random.uniform(0, 0.1))
        margin_y = int(h * random.uniform(0, 0.1))
        left = max(0, x - margin_x)
        top = max(0, y - margin_y)
        right = min(img_width, x + w + margin_x)
        bottom = min(img_height, y + h + margin_y)

    elif case == 3:  # 45.5% margin up to 25%
        margin_x = int(w * random.uniform(0, 0.25))
        margin_y = int(h * random.uniform(0, 0.25))
        left = max(0, x - margin_x)
        top = max(0, y - margin_y)
        right = min(img_width, x + w + margin_x)
        bottom = min(img_height, y + h + margin_y)

    elif case == 4:  # 16.25% one larger side margin
        large_margin = random.uniform(0.5, 1.2)
        small_margin = random.uniform(0, 0.05)
        side = random.choice(["horizontal", "vertical"])

        if side == "horizontal":
            margin_x = int(w * large_margin)
            margin_y = int(h * small_margin)
        else:
            margin_x = int(w * small_margin)
            margin_y = int(h * large_margin)

        left = max(0, x - (margin_x if side == "horizontal" else 0))
        top = max(0, y - (margin_y if side == "vertical" else 0))
        right = min(img_width, x + w + (margin_x if side == "horizontal" else 0))
        bottom = min(img_height, y + h + (margin_y if side == "vertical" else 0))

    elif case == 5:  # 8.25% both sides large margins
        margin_x = int(w * random.uniform(0.3, 0.7))
        margin_y = int(h * random.uniform(0.3, 0.7))
        left = max(0, x - margin_x)
        top = max(0, y - margin_y)
        right = min(img_width, x + w + margin_x)
        bottom = min(img_height, y + h + margin_y)

    return (int(left), int(top), int(right), int(bottom))

# Process each dataset (train/val)
for split, image_folder in image_folders.items():
    annotations_file = annotations_files[split]
    split_output_folder = os.path.join(output_folder, split)

    # Load annotations
    with open(annotations_file, "r") as f:
        data = json.load(f)

    categories = data["categories"]
    id_to_name = {category["id"]: category["name"] for category in categories}

    # Prepare new annotations
    new_annotations = []

    for img_info in tqdm(data["images"], desc=f"Processing {split} images"):
        image_id = img_info["id"]
        image_path = os.path.join(image_folder, img_info["file_name"])

        if not os.path.exists(image_path):
            continue

        img = Image.open(image_path)
        img_width, img_height = img.size

        # Get annotations for this image
        annotations = [
            ann for ann in data["annotations"] if ann["image_id"] == image_id
        ]

        for ann in annotations:
            category_id = ann["category_id"]
            category_name = id_to_name[category_id]
            bbox = ann["bbox"]

            # Generate loose bounding box
            left, top, right, bottom = generate_loose_bboxes(bbox, img_width, img_height)

            # Check if the cropped image meets the dimension requirements
            crop_width = right - left
            crop_height = bottom - top
            if min(crop_width, crop_height) < min(min_dim1, min_dim2) or max(crop_width, crop_height) < max(min_dim1, min_dim2):
                continue

            # Crop and save image
            cropped_img = img.crop((left, top, right, bottom))
            cropped_img_name = f"{image_id}_{ann['id']}.jpg"
            cropped_img_path = os.path.join(split_output_folder, cropped_img_name)
            cropped_img.save(cropped_img_path)

            # Update annotation
            new_bbox = [
                bbox[0] - left,
                bbox[1] - top,
                bbox[2],
                bbox[3]
            ]

            new_annotations.append({
                "file_name": cropped_img_name,
                "bbox": new_bbox,
                "category_id": category_id,
                "category_name": category_name,
            })

    # Save updated annotations
    output_annotations_file = os.path.join(output_folder, f"instances_{split}.json")
    with open(output_annotations_file, "w") as f:
        json.dump({"annotations": new_annotations}, f, indent=4)

print(f"New preprocessed dataset created at {output_folder}")