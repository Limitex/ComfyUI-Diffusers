import torch
from comfy.model_management import get_torch_device  # pyright: ignore[reportMissingImports]
from diffusers import AutoencoderKL, StableDiffusionPipeline
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from PIL import Image

from ...domain.repositories import SamplingRepository


class DiffusersSamplingRepository(SamplingRepository):
    def __init__(self) -> None:
        self.device = get_torch_device()

    def sample(
        self,
        pipeline: StableDiffusionPipeline,
        vae: AutoencoderKL,
        scheduler: SchedulerMixin,
        positive_embeds: torch.Tensor,
        negative_embeds: torch.Tensor,
        width: int,
        height: int,
        steps: int,
        cfg: float,
        seed: int,
    ) -> list[Image.Image]:
        result = pipeline(  # type: ignore[operator]
            prompt_embeds=positive_embeds,
            vae=vae,
            scheduler=scheduler,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=cfg,
            negative_prompt_embeds=negative_embeds,
            generator=torch.Generator(self.device).manual_seed(seed),
        )
        images: list[Image.Image] = result.images
        return images
