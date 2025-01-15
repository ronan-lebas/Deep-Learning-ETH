from PIL import Image, ImageDraw, ImageFont


def draw_bbox(image, bbox, color='red'):
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
    draw.rectangle(rectangle, outline=color, width=2)

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

def draw_label(image, label):
    """
    Draws a label (text) on the bottom-left of the image.
    
    Parameters:
    - image: PIL.Image.Image, the input image.
    - label: list of float, label in the format [center_x, center_y, width, height].
    
    Returns:
    - PIL.Image.Image, the image with the label drawn on it.
    """
    # Ensure the image is in RGB mode
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)

    # Load font
    try:
        font = ImageFont.truetype("arial.ttf", 22)  # Use a standard font
    except IOError:
        font = ImageFont.load_default(size=22)  # Fallback to default font if unavailable


    # Set text position (bottom-left of the image)
    text_x = 10  # Slight padding from the left
    text_y = image.height - 22 - 10  # Slight padding from the bottom
    
    draw.rectangle(
        [text_x - 5, text_y - 5, text_x + 330, text_y + 22 + 5], fill="black"
    )

    # Draw the text on top of the rectangle
    draw.text((text_x, text_y), label, fill="white", font=font)

    return image