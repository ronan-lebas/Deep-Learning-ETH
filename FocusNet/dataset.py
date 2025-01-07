import os
import json
import random
from PIL import Image, ImageOps
from tqdm import tqdm
from typing import Tuple
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CroppedImageDataset(Dataset):
    def __init__(self, mode: str, split: str, transform=None):
        self.base_folder = (
            "./processed_dataset"  # Hardcoded base folder for processed data
        )
        self.original_folder = (
            "./dataset/material_version/"  # Hardcoded folder for raw dataset
        )
        self.mode = mode
        self.split = "train" if split == "train" else "val"
        self.image_folder = os.path.join(self.base_folder, self.mode, self.split)
        self.annotations_file = os.path.join(
            self.base_folder, self.mode, f"instances_{self.split}_trashcan.json"
        )

        if not os.path.exists(self.image_folder) or not os.path.exists(
            self.annotations_file
        ):
            self._preprocess()

        with open(self.annotations_file, "r") as f:
            self.annotations = json.load(f)["annotations"]

        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_path = os.path.join(self.image_folder, annotation["file_name"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        bbox = torch.tensor(annotation["bbox"], dtype=torch.float32)
        label = self.bbox_to_label(
            bbox, 124, 124
        ).float()
        image_id = annotation["file_name"]

        return image, label, image_id

    def bbox_to_label(
        self, bbox: torch.Tensor, img_width: int, img_height: int
    ) -> torch.Tensor:
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
        x, y, w, h = bbox.unbind(0)

        # Compute center coordinates
        x_center = x + w / 2.0
        y_center = y + h / 2.0

        # Normalize coordinates
        x_center /= img_width
        y_center /= img_height
        w /= img_width
        h /= img_height
        
        x_center = torch.clamp(x_center, 0.0, 1.0)
        y_center = torch.clamp(y_center, 0.0, 1.0)
        w = torch.clamp(w, 0.0, 1.0)
        h = torch.clamp(h, 0.0, 1.0)

        return torch.tensor([x_center, y_center, w, h])

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

    def get_original_image(self, image_id: str) -> Image.Image:
        for annotation in self.annotations:
            if annotation["file_name"] == image_id:
                original_image_path = os.path.join(
                    self.original_folder, self.split, annotation["original_file_name"]
                )
                # crop it correctly
                crop = annotation["crop"]
                left, top, width, height = crop
                original_image = Image.open(original_image_path)
                return original_image.crop((left, top, left + width, top + height))

        raise ValueError(f"Image ID {image_id} not found.")

    def _preprocess(self):
        print(f"Preprocessing data for mode '{self.mode}' and split '{self.split}'...")

        image_folder = os.path.join(self.original_folder, self.split)
        annotations_file = os.path.join(
            self.original_folder, f"instances_{self.split}_trashcan.json"
        )

        os.makedirs(self.image_folder, exist_ok=True)

        if not os.path.exists(annotations_file):
            raise FileNotFoundError(f"Annotations file not found: {annotations_file}")

        with open(annotations_file, "r") as f:
            data = json.load(f)

        new_annotations = []

        for img_info in tqdm(
            data.get("images", []), desc=f"Processing {self.split} images"
        ):
            image_id = img_info.get("id")
            image_path = os.path.join(image_folder, img_info.get("file_name", ""))

            if not os.path.exists(image_path):
                continue

            try:
                img = Image.open(image_path)
                img_width, img_height = img.size
            except Exception as e:
                print(f"Error opening image {image_path}: {e}")
                continue

            annotations = [
                ann
                for ann in data.get("annotations", [])
                if ann.get("image_id") == image_id
            ]

            for ann in annotations:
                bbox = ann.get("bbox", [])
                if len(bbox) != 4:
                    print(f"Invalid bbox format for annotation {ann}")
                    continue

                # Generate loose bounding box
                left, top, width, height = self._generate_loose_bboxes(
                    bbox, img_width, img_height
                )

                cropped_img = img.crop((left, top, left + width, top + height))
                cropped_size = cropped_img.size

                bbox_left = bbox[0] - left
                bbox_top = bbox[1] - top
                bbox_width = bbox[2]
                bbox_height = bbox[3]

                # Initialize padding variables
                x1 = y1 = x2 = y2 = 0

                if self.mode in ["basic_padding", "advanced_padding"]:
                    final_image, (x1, y1, x2, y2) = self._pad_and_resize(
                        cropped_img, mode=self.mode.split("_")[0]
                    )
                    
                    new_left = max(bbox_left * (x2 - x1) / cropped_size[0] + x1, x1)
                    new_top = max(bbox_top * (y2 - y1) / cropped_size[1] + y1, y1)
                    new_width = bbox_width * (x2 - x1) / cropped_size[0]
                    if(new_width + new_left > x2):
                        new_width = x2 - new_left
                    new_height = bbox_height * (y2 - y1) / cropped_size[1]
                    if(new_height + new_top > y2):
                        new_height = y2 - new_top
                    
                    new_bbox = [
                        new_left,
                        new_top,
                        new_width,
                        new_height,
                    ]

                elif self.mode == "basic":
                    final_image = cropped_img.resize((124, 124), Image.LANCZOS)
                    # Recompute the bbox coordinates depending on the new size
                    new_bbox = [
                        bbox_left * 124 / cropped_size[0],
                        bbox_top * 124 / cropped_size[1],
                        bbox_width * 124 / cropped_size[0],
                        bbox_height * 124 / cropped_size[1],
                    ]

                elif self.mode == "filtering":
                    crop_width, crop_height = cropped_img.size
                    min_dim1, min_dim2 = 90, 40
                    if min(crop_width, crop_height) < min(min_dim1, min_dim2) or max(
                        crop_width, crop_height
                    ) < max(min_dim1, min_dim2):
                        continue
                    final_image = cropped_img.resize((124, 124), Image.LANCZOS)
                    new_bbox = [
                        bbox_left * 124 / cropped_size[0],
                        bbox_top * 124 / cropped_size[1],
                        bbox_width * 124 / cropped_size[0],
                        bbox_height * 124 / cropped_size[1],
                    ]
                else:
                    print(f"Unsupported mode: {self.mode}")
                    continue

                final_image_name = f"{image_id}_{ann['id']}.jpg"
                final_image_path = os.path.join(self.image_folder, final_image_name)

                try:
                    final_image.save(final_image_path)
                except Exception as e:
                    print(f"Error saving cropped image {final_image_path}: {e}")
                    continue

                new_annotations.append(
                    {
                        "file_name": final_image_name,
                        "bbox": new_bbox,
                        "category_id": ann.get("category_id"),
                        "width": cropped_size[0],
                        "height": cropped_size[1],
                        "padding": (
                            (x1, y1, x2, y2)
                            if self.mode in ["basic_padding", "advanced_padding"]
                            else None
                        ),
                        "crop": (left, top, width, height),
                        "original_file_name": img_info.get("file_name"),
                        "original_bbox": [bbox_left, bbox_top, bbox_width, bbox_height],
                    }
                )

        with open(self.annotations_file, "w") as f:
            json.dump({"annotations": new_annotations}, f, indent=4)

    @staticmethod
    def _generate_loose_bboxes(
        bbox: Tuple[int, int, int, int], img_width: int, img_height: int
    ) -> Tuple[int, int, int, int]:
        x, y, w, h = bbox
        case = random.choices(
            [1, 2, 3, 4, 5], weights=[15, 20, 45.5, 16.25, 8.25], k=1
        )[0]

        if case == 1:
            left, top, right, bottom = int(x), int(y), int(x + w), int(y + h)

        elif case == 2:
            margin_x = int(w * random.uniform(0, 0.1))
            margin_y = int(h * random.uniform(0, 0.1))
            left = max(0, x - margin_x)
            top = max(0, y - margin_y)
            right = min(img_width, x + w + margin_x)
            bottom = min(img_height, y + h + margin_y)

        elif case == 3:
            margin_x = int(w * random.uniform(0, 0.25))
            margin_y = int(h * random.uniform(0, 0.25))
            left = max(0, x - margin_x)
            top = max(0, y - margin_y)
            right = min(img_width, x + w + margin_x)
            bottom = min(img_height, y + h + margin_y)

        elif case == 4:
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

        elif case == 5:
            margin_x = int(w * random.uniform(0.3, 0.7))
            margin_y = int(h * random.uniform(0.3, 0.7))
            left = max(0, x - margin_x)
            top = max(0, y - margin_y)
            right = min(img_width, x + w + margin_x)
            bottom = min(img_height, y + h + margin_y)

        width = right - left
        height = bottom - top
        return int(left), int(top), int(width), int(height)

    @staticmethod
    def _pad_and_resize(
        image: Image.Image, target_size: int = 124, mode: str = "basic"
    ) -> Image.Image:
        img_width, img_height = image.size
        if mode == "basic":
            if max(img_width, img_height) > target_size:
                image.thumbnail((target_size, target_size))
                img_width, img_height = image.size
            delta_w = target_size - image.size[0]
            delta_h = target_size - image.size[1]
            padding = (
                delta_w // 2,
                delta_h // 2,
                delta_w - (delta_w // 2),
                delta_h - (delta_h // 2),
            )
            x1, y1 = delta_w // 2, delta_h // 2
            x2, y2 = x1 + img_width, y1 + img_height
            return ImageOps.expand(image, padding, fill="black"), (x1, y1, x2, y2)

        elif mode == "advanced":
            scale_factor = target_size / max(img_width, img_height)
            new_size = (int(img_width * scale_factor), int(img_height * scale_factor))
            image = image.resize(new_size, Image.LANCZOS)
            delta_w = target_size - new_size[0]
            delta_h = target_size - new_size[1]
            padding = (
                delta_w // 2,
                delta_h // 2,
                delta_w - (delta_w // 2),
                delta_h - (delta_h // 2),
            )
            x1, y1 = delta_w // 2, delta_h // 2
            x2, y2 = x1 + new_size[0], y1 + new_size[1]
            return ImageOps.expand(image, padding, fill="black"), (x1, y1, x2, y2)

        raise ValueError(f"Unknown mode: {mode}")
