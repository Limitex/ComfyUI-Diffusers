import torch
from comfy.model_management import get_torch_device  # pyright: ignore[reportMissingImports]
from diffusers import AutoencoderKL, StableDiffusionPipeline
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from PIL import Image

from ...domain.model import CFGScale, ImageSize, Seed, Steps
from ...domain.repositories import SamplerRepository


class DiffusersSamplerRepository(SamplerRepository):
    def __init__(self) -> None:
        self.device = get_torch_device()

    def sample(
        self,
        pipeline: StableDiffusionPipeline,
        vae: AutoencoderKL,
        scheduler: SchedulerMixin,
        positive_embeds: torch.Tensor,
        negative_embeds: torch.Tensor,
        image_size: ImageSize,
        steps: Steps,
        cfg: CFGScale,
        seed: Seed,
    ) -> list[Image.Image]:
        pipeline.vae = vae  # type: ignore[attr-defined]
        pipeline.scheduler = scheduler  # type: ignore[attr-defined]
        result = pipeline.to(self.device)(  # type: ignore[attr-defined]
            prompt_embeds=positive_embeds,
            height=image_size.height,
            width=image_size.width,
            num_inference_steps=steps.value,
            guidance_scale=cfg.value,
            negative_prompt_embeds=negative_embeds,
            generator=torch.Generator(self.device).manual_seed(seed.value),
        )
        images: list[Image.Image] = result.images
        return images
