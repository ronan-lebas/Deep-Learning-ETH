from dataset.trashcan_dataset import TrashCanDataset
from augmentation.augmentation_utils import AugmentationUtils
from models.inpainting_model import InpaintingPipeline
from models.focus_net import FocusNet
import os
from tqdm import tqdm
from PIL import Image
import json

class AugmentationPipeline:
    def __init__(
        self, dataset_path: str, image_dir: str, inpainting_model_path: str, lora_paths: list, focus_net_path: str
    ):
        self.dataset = TrashCanDataset(dataset_path, image_dir)
        self.inpainting = InpaintingPipeline(inpainting_model_path)
        self.inpainting.load_lora_weights(lora_paths[0])  # Load first LoRA for simplicity
        self.focus_net = FocusNet(focus_net_path)

    def generate_augmented_images(self, output_dir: str, num_images: int = 10000):
        os.makedirs(output_dir, exist_ok=True)

        for i in tqdm(range(num_images), desc="Generating images"):
            # random image ID from the dataset
            image_id = self.dataset.get_random_image_id()
            image = self.dataset.load_image(image_id)

            # random bounding box and mask (todo)
            bbox, category = self.dataset.get_random_bbox_and_category(image_id)
            mask, mask_coords = AugmentationUtils.generate_mask(image.size, bbox)

            # Apply inpainting with the model
            inpainted_image = self.inpainting.apply_inpainting(
                prompt="a piece of trash floating underwater, realistic, blend with the environment",
                negative_prompt="text, clean, clear, nothing",
                init_image=image,
                mask_image=mask
            )

            # Refine the bounding box with FocusNet
            refined_bbox = self.focus_net.refine_bounding_box(inpainted_image, bbox)

            # Save results
            output_image_path = os.path.join(output_dir, f"image_{i+1:05d}.png")
            output_metadata_path = os.path.join(output_dir, f"image_{i+1:05d}_meta.json")

            inpainted_image.save(output_image_path)

            metadata = {
                "image_id": image_id,
                "original_bbox": bbox,
                "refined_bbox": refined_bbox,
                "category": category
            }

            with open(output_metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)

        print(f"Augmented images and metadata saved to {output_dir}")
