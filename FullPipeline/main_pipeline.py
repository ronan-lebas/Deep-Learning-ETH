from dataset.trashcan_dataset import TrashCanDataset
from utils.visualization_utils import draw_bbox, draw_mask, draw_label
from augmentation.augmentation_utils import generate_mask, is_image_convenient
from models.inpainting_model import InpaintingPipeline

# I put this version for the moment bc i don't have Linus weights
from models.focus_net import FocusNetv4
import os
from tqdm import tqdm
from PIL import Image
import json
from config import *
import torch
import re
import random

class AugmentationPipeline:
    def __init__(
        self,
        dataset_path: str,
        augmented_dir: str,
        lora_path: str,
        lora_name: str,
        focus_net_path: str,
        excluded_categories: list = [],
        replaced_categories: dict = {},
    ):
        self.augmented_dir = augmented_dir
        self.excluded_categories = excluded_categories
        self.replaced_categories = replaced_categories
        
        self.dataset = TrashCanDataset(os.path.join(dataset_path, "instances_train_trashcan.json"), os.path.join(dataset_path,"train"))
        self.inpainting = InpaintingPipeline()
        self.inpainting.load_lora_weights(lora_path, lora_name)
        self.focus_net = FocusNetv4()
        self.focus_net.load_state_dict(torch.load(focus_net_path, weights_only=True))
        self.focus_net.eval()
        
        self.prompts = prompts_v2

    def generate_one_augmented_image(self):
        """
        Generates one augmented image and saves it to the output directory.
        return: the image, the metadata
        """

        # random image ID from the dataset
        image_id = self.dataset.get_random_image_id()
        bbox, category = self.dataset.get_random_bbox_and_category(image_id)
        image = self.dataset.load_image(image_id)
        while not is_image_convenient(image, bbox, 50, 50) or category in self.excluded_categories:
            image_id = self.dataset.get_random_image_id()
            bbox, category = self.dataset.get_random_bbox_and_category(image_id)
            image = self.dataset.load_image(image_id)
            
        if category in self.replaced_categories:
            category = random.choice(self.replaced_categories[category])
        
        print(f"Image ID: {image_id}, Category: {category}")
        
        mask = generate_mask(image, bbox)

        #Apply inpainting with the model
        inpainted_image = self.inpainting.apply_inpainting(
            prompt=self.prompts[category],
            negative_prompt="text, clean, clear, nothing",
            init_image=image,
            mask_image=mask,
            strength=0.975,
            padding_mask_crop=64,
            num_inference_steps=50,
        )

        # Refine the bounding box with FocusNet
        # this is my (maxime) implementatiuon, can be swapped with linus one just by changing the import
        refined_bbox = self.focus_net.refine_bounding_box(inpainted_image, bbox)

        metadata = {
            "image_id": image_id,
            "original_bbox": bbox,
            "refined_bbox": refined_bbox,
            "category": category,
        }

        return image, inpainted_image, metadata

    def generate_augmented_images(self, num_images: int = 10000, augmented_dir: str = None):
        output_dir = augmented_dir if augmented_dir is not None else self.augmented_dir
        os.makedirs(output_dir, exist_ok=True)

        for i in tqdm(range(num_images), desc="Generating images"):
            try:
                original_image, inpaited_image, metadata = self.generate_one_augmented_image()
            except ValueError as e:
                print(f"Error: {e}")
                continue
            # Save the image and metadata
            inpaited_image.save(os.path.join(output_dir, f"image_{i}_{metadata['image_id']}.png"))
            # add the fileanme to metadata
            metadata["filename"] = f"image_{i}_{metadata['image_id']}.png"
            with open(os.path.join(output_dir, f"metadata_{i}.json"), "w") as f:
                json.dump(metadata, f)
                
            # for the report
            inpaited_image = draw_bbox(inpaited_image, metadata["original_bbox"])
            inpaited_image = draw_bbox(inpaited_image, metadata["refined_bbox"], color='yellow')
            inpaited_image = draw_label(inpaited_image, category_labels[metadata["category"]])
            inpaited_image.save(os.path.join("for_the_report", f"image_{i}.png"))
            original_image = draw_bbox(original_image, metadata["original_bbox"])
            original_image.save(os.path.join("for_the_report", f"image_{i}_original.png"))            
        
        # merge all the json into one file
        metadata = []
        for filename in os.listdir(output_dir):
        # Check if the filename matches the pattern "metadata_*.json"
            if re.match(r"metadata_\d+\.json", filename):
                file_path = os.path.join(output_dir, filename)
                with open(file_path, "r") as f:
                    metadata.append(json.load(f))

        # Write the merged metadata to a single JSON file
        output_file = os.path.join(output_dir, "augmented_trashcan.json")
        with open(output_file, "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"Merged metadata written to {output_file}")


if __name__ == "__main__":

    pipeline = AugmentationPipeline(
        DATASET_PATH,
        AUGMENTED_DATASET_PATH,
        LORA_PATH,
        LORA_WEIGHTS,
        FOCUS_NET_PATH,
        excluded_categories=[1], # The rover is not relevant for the augmentation
        replaced_categories={17: [9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22]} # Replace the unknown category with the trash categories
    )
    
    pipeline.generate_augmented_images(5)