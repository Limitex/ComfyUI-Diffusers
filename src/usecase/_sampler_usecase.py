from __future__ import annotations

import torch

from ..domain.model import (
    Autoencoder,
    CFGScale,
    Conditioning,
    Image,
    ImageSize,
    Pipeline,
    Scheduler,
    Seed,
    Steps,
)
from ..domain.repositories import SamplerRepository


class SamplerUsecase:
    def __init__(self, sampler_repo: SamplerRepository) -> None:
        self.sampler_repo = sampler_repo
        self.dtype = torch.float16

    def execute(
        self,
        pipeline: Pipeline,
        vae: Autoencoder,
        scheduler: Scheduler,
        positive_embeds: Conditioning,
        negative_embeds: Conditioning,
        image_size: ImageSize,
        steps: Steps,
        cfg: CFGScale,
        seed: Seed,
    ) -> list[Image]:
        images = self.sampler_repo.sample(
            pipeline.pipeline,
            vae.autoencoder,
            scheduler.scheduler,
            positive_embeds.conditioning,
            negative_embeds.conditioning,
            image_size,
            steps,
            cfg,
            seed,
        )

        if not images:
            raise RuntimeError("Sampler repository returned no images.")

        return [Image(image=img) for img in images]
