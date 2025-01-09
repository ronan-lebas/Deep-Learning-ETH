import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
from torchvision import transforms
from focusnet import FocusNet, FocusNetDataset


def eval(expansion_ratio, device,model,weights):
    # Device setup

    ANNOTATION_FILE = 'instances_train_trashcan_expanded.json'
    IMAGE_DIR = 'train'
    OUTPUT_PDF = 'annotated_images_inference.pdf'
    NUM_IMAGES = 56  # Total number of images to include in the PDF
    NUM_IMAGES_PER_PAGE = 8

    # Dataset and Model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = FocusNetDataset(
        annotation_file=ANNOTATION_FILE,
        image_dir=IMAGE_DIR,
        transform=transform,
        expansion_ratio=expansion_ratio
    )

    # Load Pre-trained Model
    model.load_state_dict(torch.load(weights, weights_only=True, map_location=device))
    model.eval()

    def denormalize_bbox(bbox, img_width, img_height):
        x, y, w, h = bbox
        return [
            x * img_width,
            y * img_height,
            w * img_width,
            h * img_height
        ]

    # Create a PDF file
    with PdfPages(OUTPUT_PDF) as pdf:
        for page_start in range(0, min(NUM_IMAGES, len(dataset)), NUM_IMAGES_PER_PAGE):  # 6 images per page
            fig, axes = plt.subplots(NUM_IMAGES_PER_PAGE//2, 2, figsize=(16, 24))  # 3 rows x 2 columns
            axes = axes.flatten()

            for i in range(NUM_IMAGES_PER_PAGE):  # Process 6 images per page
                idx = page_start + i
                if idx >= len(dataset):
                    axes[i].axis('off')
                    continue

                with torch.no_grad():
                    # Fetch dataset sample
                    image, expanded_bbox, original_bbox = dataset[idx]

                    # Load raw image (untransformed) for visualization
                    raw_image_path = os.path.join(IMAGE_DIR, dataset.annotations[idx]['file_name'])
                    raw_image = Image.open(raw_image_path).convert("RGB")
                    img_width, img_height = raw_image.size  # Get raw image dimensions

                    # Move tensors to the correct device
                    image = image.unsqueeze(0).to(device)
                    expanded_bbox = expanded_bbox.unsqueeze(0).to(device)
                    original_bbox = original_bbox.to(device)

                    # Perform inference
                    predicted_bbox = model(image, expanded_bbox)

                    # Extract BBox Coordinates
                    expanded_bbox = expanded_bbox.squeeze(0).cpu().numpy()
                    predicted_bbox = predicted_bbox.squeeze(0).cpu().numpy()
                    original_bbox = original_bbox.cpu().numpy()

                    expanded_bbox = denormalize_bbox(expanded_bbox, img_width, img_height)
                    predicted_bbox = denormalize_bbox(predicted_bbox, img_width, img_height)
                    original_bbox = denormalize_bbox(original_bbox, img_width, img_height)

                # Visualization on Subplot
                axes[i].imshow(raw_image)
                axes[i].set_title(f"Image: {dataset.annotations[idx]['file_name']}", fontsize=8)
                axes[i].axis('off')

                # 🟦 Draw Expanded Bounding Box (Blue)
                x, y, w, h = expanded_bbox
                expanded_rect = patches.Rectangle(
                    (x, y), w, h,
                    linewidth=2, edgecolor='blue', facecolor='none', label='Expanded BBox'
                )
                axes[i].add_patch(expanded_rect)

                # 🟥 Draw Ground Truth Bounding Box (Red)
                x, y, w, h = original_bbox
                original_rect = patches.Rectangle(
                    (x, y), w, h,
                    linewidth=2, edgecolor='red', facecolor='none', label='Ground Truth BBox'
                )
                axes[i].add_patch(original_rect)

                # 🟩 Draw Predicted Bounding Box (Green)
                x, y, w, h = predicted_bbox
                predicted_rect = patches.Rectangle(
                    (x, y), w, h,
                    linewidth=2, edgecolor='green', facecolor='none', label='Predicted BBox'
                )
                axes[i].add_patch(predicted_rect)

                # 📝 Add Legend
                axes[i].legend(loc='upper right', fontsize=6)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"PDF with {NUM_IMAGES} annotated images (6 per page) saved as '{OUTPUT_PDF}'")
