from diffusers import AutoPipelineForInpainting
import torch

class InpaintingPipeline:
    def __init__(self, model_path: str):
        self.pipeline = AutoPipelineForInpainting.from_pretrained(
            model_path, torch_dtype=torch.float16, variant="fp16"
        )

    def load_lora_weights(self, lora_path: str):
        self.pipeline.load_lora_weights(lora_path)
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
