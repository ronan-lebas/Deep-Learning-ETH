from dataset.trashcan_dataset import TrashCanDataset
from utils.visualization_utils import draw_bbox, draw_mask
from augmentation.augmentation_utils import generate_mask
from models.inpainting_model import InpaintingPipeline

# I put this version for the moment bc i don't have Linus weights
from models.focus_net import FocusNetv4
import os
from tqdm import tqdm
from PIL import Image
import json
from config import *
import torch


class AugmentationPipeline:
    def __init__(
        self,
        dataset_path: str,
        image_dir: str,
        lora_path: str,
        lora_name: str,
        focus_net_path: str,
    ):
        self.dataset = TrashCanDataset(os.path.join(dataset_path, "instances_train_trashcan.json"), os.path.join(dataset_path,"train"))
        self.inpainting = InpaintingPipeline()
        self.focus_net = FocusNetv4()
        self.focus_net.load_state_dict(torch.load(focus_net_path), weights_only=True)
        self.focus_net.eval()

    def generate_one_augmented_image(self):
        """
        Generates one augmented image and saves it to the output directory.
        return: the image, the metadata
        """

        # random image ID from the dataset
        image_id = self.dataset.get_random_image_id()
        image = self.dataset.load_image(image_id)

        # random bounding box and mask (todo)
        bbox, category = self.dataset.get_random_bbox_and_category(image_id)
        mask = generate_mask(image, bbox)

        #Apply inpainting with the model
        inpainted_image = self.inpainting.apply_inpainting(
            prompt="a piece of trash floating underwater, realistic, blend with the environment",
            negative_prompt="text, clean, clear, nothing",
            init_image=image,
            mask_image=mask
        )

        # Refine the bounding box with FocusNet
        # refined_bbox = self.focus_net.refine_bounding_box(inpainted_image, bbox)

        metadata = {
            "image_id": image_id,
            "original_bbox": bbox,
            "refined_bbox": None,
            "category": category,
        }

        return image, inpainted_image, mask, metadata

    def generate_augmented_images(self, output_dir: str, num_images: int = 10000):
        os.makedirs(output_dir, exist_ok=True)

        for i in tqdm(range(num_images), desc="Generating images"):
            image, masked_image, metadata = self.generate_one_augmented_image()

            # Save the image and metadata
            image.save(os.path.join(output_dir, f"image_{i}.png"))
            masked_image.save(os.path.join(output_dir, f"masked_image_{i}.png"))
            with open(os.path.join(output_dir, f"metadata_{i}.json"), "w") as f:
                json.dump(metadata, f)


if __name__ == "__main__":

    pipeline = AugmentationPipeline(
        DATASET_PATH,
        AUGMENTED_DATASET_PATH,
        LORA_PATH,
        LORA_WEIGHTS,
        FOCUS_NET_PATH,
    )
    image, inpainted_image, mask, metadata = pipeline.generate_one_augmented_image()
    
    bbox_image = draw_bbox(image, metadata["original_bbox"])

    # Save the image and metadata
    image.save("output_image.png")
    inpainted_image.save("output_inpainted_image.png")
    bbox_image.save("output_bbox_image.png")
    mask.save("output_mask.png")
    
    print(metadata)
