import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid

pipeline = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, variant="fp16"
)
pipeline.enable_model_cpu_offload()

# load base and mask image
init_image = load_image("test_images/back.jpg")
mask_image = load_image("test_images/mask1.png")
# resize them 512x512
init_image = init_image.resize((512, 512))
mask_image = mask_image.resize((512, 512))

generator = torch.Generator("cuda").manual_seed(92)
prompt = "a plastic bag on the ocean"
image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, strength=0.8, generator=generator, padding_mask_crop=32).images[0]
image_grid = make_image_grid([init_image, mask_image, image], rows=1, cols=3)

image_grid.save("inpainting.png")

