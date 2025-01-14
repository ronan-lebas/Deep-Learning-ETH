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
   