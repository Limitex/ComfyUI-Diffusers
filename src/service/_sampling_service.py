import torch
from diffusers import StableDiffusionPipeline

from ..domain.model import Image
from ..domain.repositories import SamplingRepository


class SamplingService:
    def __init__(self, sampling_repo: SamplingRepository) -> None:
        self.sampling_repo = sampling_repo
        self.dtype = torch.float32

    def sample(
        self,
        pipeline: StableDiffusionPipeline,
        positive_embeds: torch.Tensor,
        negative_embeds: torch.Tensor,
        width: int,
        height: int,
        steps: int,
        cfg: float,
        seed: int,
    ) -> list[Image]:
        images = self.sampling_repo.sample(
            pipeline, positive_embeds, negative_embeds, width, height, steps, cfg, seed
        )
        if images is None:
            raise RuntimeError("Sampling repository returned no images.")
        images_domain: list[Image] = [Image(image=img) for img in images]
        return images_domain
