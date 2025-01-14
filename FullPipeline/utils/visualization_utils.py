from PIL import Image, ImageDraw


def draw_bbox(image, bbox):
    """
    Draws a bounding box on an image.

    Parameters:
    - image: PIL.Image.Image, the input image.
    - bbox: list of float, bounding box in the format [x, y, width, height].

    Returns:
    - PIL.Image.Image, the image with the bounding box drawn on it.
    """
    # Ensure bbox has the required format
    if len(bbox) != 4:
        raise ValueError(
            "Bounding box must be a list of 4 floats: [x, y, width, height]"
        )
        
    image_copy = image.copy()

    # Unpack bounding box
    x, y, width, height = bbox

    # Create a draw object
    draw = ImageDraw.Draw(image_copy)

    # Define the bounding box as a rectangle
    rectangle = [x, y, x + width, y + height]

    # Draw the rectangle (bounding box)
    draw.rectangle(rectangle, outline="red", width=2)

    return image_copy

def draw_mask(image, bbox):
    """
    Draws a mask on an image.

    Parameters:
    - image: PIL.Image.Image, the input image.
    - bbox: list of float, bounding box in the format [x, y, width, height].

    Returns:
    - PIL.Image.Image, the image with the mask drawn on it.
    """
    # Ensure bbox has the required format
    if len(bbox) != 4:
        raise ValueError(
            "Bounding box must be a list of 4 floats: [x, y, width, height]"
        )

    image_copy = image.copy()

    # Unpack bounding box
    x, y, width, height = bbox

    # Create a draw object
    draw = ImageDraw.Draw(image_copy)

    # Define the bounding box as a rectangle
    rectangle = [x, y, x + width, y + height]

    # Draw the a black mask
    draw.rectangle(rectangle, fill="black")

    return image_copy