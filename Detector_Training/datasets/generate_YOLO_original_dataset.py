import os
import json
import argparse
import random
from tqdm import tqdm
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
#from torchvision import transforms

class ImagePreprocessor:
    #mode is either material_version or instance_version (we use the 1st one)
    def __init__(self, split: str, path_to_dataset='.', transform=None):
        self.base_folder = (
            path_to_dataset+f"/original_dataset_YOLO/"  # Hardcoded base folder for processed data
        )
        self.original_folder = (
            path_to_dataset+f"/original_dataset/"  # Hardcoded folder for raw dataset
        )
        self.split = "train" if split == "train" else "val"
        self.images_folder = os.path.join(self.base_folder, self.split, "images")
        self.labels_folder = os.path.join(self.base_folder, self.split, "labels")
        #self.annotations_file = os.path.join(
        #    self.base_folder, self.mode, f"instances_{self.split}_trashcan.json"
        #)

        if not os.path.exists(self.images_folder) or not os.path.exists(
                self.labels_folder
        ):
            self._preprocess()


    def bbox_to_label(
            self, bbox: list, img_width: int, img_height: int
    ) -> np.array:
        """
        Convert bounding box coordinates to YOLO label format.

        Args:
            bbox (torch.Tensor): Bounding box in (x, y, w, h) format where x, y are the top-left coordinates.
            img_width (int): Width of the image.
            img_height (int): Height of the image.

        Returns:
            torch.Tensor: Bounding box in YOLO format (center_x, center_y, width, height),
                        normalized by the image dimensions.
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


    def _preprocess(self):
        print(f"Preprocessing data for split '{self.split}'...")

        image_folder = os.path.join(self.original_folder, self.split)
        annotations_file = os.path.join(
            self.original_folder, f"instances_{self.split}_trashcan.json"
        )

        os.makedirs(self.images_folder, exist_ok=True)
        os.makedirs(self.labels_folder, exist_ok=True)
        if not os.path.exists(annotations_file):
            raise FileNotFoundError(f"Annotations file not found: {annotations_file}")

        with open(annotations_file, "r") as f:
            data = json.load(f)

        categories_counts = np.zeros(30)

        for img_info in tqdm(
                data.get("images", []), desc=f"Processing {self.split} images"
        ):
            image_id = img_info.get("id")
            image_path = os.path.join(image_folder, img_info.get("file_name", ""))

            if not os.path.exists(image_path):
                print("The image path doesn't exist.")
                continue
            
            # copy the image
            os.system(f'cp {image_path} {self.images_folder}')

            img_width, img_height = img_info.get("width"), img_info.get("height")

            annotations = [
                ann
                for ann in data.get("annotations", [])
                if ann.get("image_id") == image_id
            ]

            filename = os.path.join(self.labels_folder, img_info.get("file_name").split('.')[0] + '.txt')
            with open(filename, 'w') as annotations_file:
                for ann in annotations:
                    bbox = ann.get("bbox", [])
                    if len(bbox) != 4:
                        print(f"Invalid bbox format for annotation {ann}")
                        continue

                    label = self.bbox_to_label(bbox, img_width, img_height)
                    category_id = ann.get("category_id")-1
                    categories_counts[category_id] +=1
                    annotations_file.write(str(category_id) + ''.join(' ' + str(coord) for coord in label)+'\n')

        print(categories_counts)


if __name__ == '__main__':
    """
    arguments:
    mode: material_version OR instance_version. DEFAULT: material_version
    split: train OR val OR both
    path: <path to the folder one layer above dataset>. DEFAULT: ..
    """
    print(f"Current directory: {os.getcwd()}")
    dataset = ImagePreprocessor("test", ".")
