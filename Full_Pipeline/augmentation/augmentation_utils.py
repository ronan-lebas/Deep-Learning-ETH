from PIL import Image, ImageDraw

def generate_mask(image, bbox):
    """
    Draws a mask on a black image of same size as input image.

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

    # Create a black image with the same size as the input image
    mask = Image.new("L", image.size, 0)

    # Unpack bounding box
    x, y, width, height = bbox

    # Create a draw object
    draw = ImageDraw.Draw(mask)

    # Define the bounding box as a rectangle
    rectangle = [x, y, x + width, y + height]

    # Draw the rectangle (bounding box)
    draw.rectangle(rectangle, fill=255)

    return mask


def square_padding(image, fill_color=(0, 0, 0)):
    """Adds padding to make the image square.

    Args:
        image (PIL.Image.Image): Image to be padded.
        fill_color (tuple, optional): Color to be used for padding. Defaults to (0, 0, 0).
    """
    
    width, height = image.size
    max_dim = max(width, height)
    padded_image = Image.new("RGB", (max_dim, max_dim), fill_color)
    padded_image.paste(image, ((max_dim - width) // 2, (max_dim - height) // 2))
    return padded_image

def is_image_convenient(image, bbox, min_width, min_height):
    """
    Checks if the bounding box is convenient based on specified conditions.

    Args:
        image: a PIL Image object.
        bbox: tuple (left, top, width, height) representing the bounding box.
        min_width: Minimum allowable width for the bounding box.
        min_height: Minimum allowable height for the bounding box.

    Returns:
        bool: True if the bounding box meets the criteria, False otherwise.
    """
    # Extract image dimensions
    img_width, img_height = image.size

    # Extract bounding box dimensions
    left, top, width, height = bbox

    # Condition 1: Ensure bbox dimensions are greater than the minimum required
    if width < min_width or height < min_height:
        print(f"Bounding box dimensions are too small: {width}x{height}")
        return False

    # Condition 2: Ensure bbox does not cover nearly the entire image
    max_coverage_width = 0.95 * img_width
    max_coverage_height = 0.95 * img_height

    if width > max_coverage_width or height > max_coverage_height:
        print(f"Bounding box dimensions are too large: {width}x{height}")
        return False

    return True