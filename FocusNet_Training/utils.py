import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

def display_resized_image_with_bbox(dataset, idx, save_path="bbox.png"):
    """Display resized image and bounding box"""
    image, bbox = dataset[idx]  # Fetch the resized image and normalized bbox (center x, center y, w, h)

    # Undo normalization to plot it as a square
    bbox_x, bbox_y, bbox_w, bbox_h = bbox.numpy()  # Extract bbox coordinates in normalized format

    # Scale back bbox for visualization on the 124x124 resized image
    image_size = 124  # Image is resized to 124x124 pixels
    xmin = (bbox_x - bbox_w / 2) * image_size
    ymin = (bbox_y - bbox_h / 2) * image_size
    xmax = (bbox_x + bbox_w / 2) * image_size
    ymax = (bbox_y + bbox_h / 2) * image_size

    # Convert the image tensor back to PIL image for visualization
    pil_image = F.to_pil_image(image)

    # Plot the image with the bbox
    plt.figure(figsize=(6, 6))
    plt.imshow(pil_image)
    plt.gca().add_patch(plt.Rectangle(
        (xmin, ymin),  # Bottom-left corner
        xmax - xmin, ymax - ymin,  # Width and Height
        fill=False, edgecolor='red', linewidth=2  # Bounding box style
    ))
    plt.title(f"BBox: x={bbox_x:.2f}, y={bbox_y:.2f}, w={bbox_w:.2f}, h={bbox_h:.2f}")
    plt.axis("off")
    plt.savefig(save_path)
    print(f"Saved visualization at {save_path}")
