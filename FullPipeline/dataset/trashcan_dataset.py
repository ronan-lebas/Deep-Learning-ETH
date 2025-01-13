from torch.utils.data import Dataset
import json
import os
from PIL import Image
import random

class TrashCanDataset(Dataset):
    def __init__(self, annotation_file, image_dir, transform=None, expansion_ratio=0.2):
        """
        Initialize the TrashCanDataset.

        :param annotation_file: Path to the JSON annotation file.
        :param image_dir: Path to the directory containing the images.
        :param transform: Transformations to apply to the images (default: None).
        :param expansion_ratio: Ratio to expand bounding boxes (default: 0.2).
        """
        with open(annotation_file, 'r') as f:
            data = json.load(f)

        self.images = {img['id']: img for img in data['images']}
        self.annotations = {}

        # Organize annotations by image ID
        for ann in data['annotations']:
            image_id = ann['image_id']
            if image_id in self.images:
                if image_id not in self.annotations:
                    self.annotations[image_id] = []
                self.annotations[image_id].append({
                    'bbox': ann['bbox'],
                    'category_id': ann['category_id']
                })

        self.image_dir = image_dir
        self.transform = transform
        self.expansion_ratio = expansion_ratio

    def __len__(self):
        """Return the number of unique images."""
        return len(self.images)

    def __getitem__(self, index):
        """Return the image ID corresponding to the given index."""
        image_ids = list(self.images.keys())
        return image_ids[index]

    def load_image(self, image_id):
        """
        Load an image given its ID.

        :param image_id: The ID of the image to load.
        :return: A PIL image.
        """
        if image_id not in self.images:
            raise ValueError(f"Image ID {image_id} not found in dataset.")

        file_name = self.images[image_id]['file_name']
        image_path = os.path.join(self.image_dir, file_name)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file {image_path} not found.")

        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

    def get_all_bboxes_and_categories(self, image_id):
        """
        Get all bounding box and category pairs for a given image ID.

        :param image_id: The ID of the image.
        :return: A list of dictionaries with 'bbox' and 'category_id'.
        """
        if image_id not in self.annotations:
            raise ValueError(f"No annotations found for image ID {image_id}.")

        return self.annotations[image_id]

    def get_random_bbox_and_category(self, image_id):
        """
        Get a random bounding box and category pair for a given image ID.

        :param image_id: The ID of the image.
        :return: A dictionary with 'bbox' and 'category_id'.
        """
        if image_id not in self.annotations:
            raise ValueError(f"No annotations found for image ID {image_id}.")

        return random.choice(self.annotations[image_id])