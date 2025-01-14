from diffusers import AutoPipelineForInpainting
import torch

class InpaintingPipeline:
    def __init__(self):
        self.pipeline = AutoPipelineForInpainting.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float16, variant="fp16"
        )
        self.pipeline.to("cuda")

    def load_lora_weights(self, model_path: str, model_name: str):
        self.pipeline.load_lora_weights(model_path, weight_name=model_name)
        self.pipeline.enable_model_cpu_offload()

    def apply_inpainting(self, prompt: str, negative_prompt: str, init_image, mask_image):
        generator = torch.Generator("cuda").manual_seed(92)
        return self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            mask_image=mask_image,
            strength=1,
            generator=generator
        ).images[0]
