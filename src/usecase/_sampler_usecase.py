import torch

from ..domain.model import Autoencoder, Conditioning, Image, Pipeline, Scheduler
from ..domain.repositories import SamplerRepository


class SamplerUsecase:
    def __init__(self, sampler_repo: SamplerRepository) -> None:
        self.sampler_repo = sampler_repo
        self.dtype = torch.float32

    def execute(
        self,
        pipeline: Pipeline,
        vae: Autoencoder,
        scheduler: Scheduler,
        positive_embeds: Conditioning,
        negative_embeds: Conditioning,
        width: int,
        height: int,
        steps: int,
        cfg: float,
        seed: int,
    ) -> list[Image]:
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid dimensions: {width}x{height}")
        if steps <= 0:
            raise ValueError(f"Steps must be positive: {steps}")
        if cfg < 0.0:
            raise ValueError(f"CFG must be non-negative: {cfg}")
        if seed < 0:
            raise ValueError(f"Seed must be non-negative: {seed}")

        images = self.sampler_repo.sample(
            pipeline.pipeline,
            vae.autoencoder,
            scheduler.scheduler,
            positive_embeds.conditioning,
            negative_embeds.conditioning,
            width,
            height,
            steps,
            cfg,
            seed,
        )
        if images is None:
            raise RuntimeError("Sampler repository returned no images.")
        images_domain: list[Image] = [Image(image=img) for img in images]

        return images_domain
