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
        if pipeline is None:
            raise ValueError("Pipeline is None.")
        if vae is None:
            raise ValueError("VAE is None.")
        if scheduler is None:
            raise ValueError("Scheduler is None.")
        if positive_embeds is None or negative_embeds is None:
            raise ValueError("Conditioning embeddings are None.")
        if width <= 0 or height <= 0:
            raise ValueError("Width and Height must be positive integers.")
        if steps <= 0:
            raise ValueError("Steps must be a positive integer.")
        if cfg < 0.0:
            raise ValueError("CFG must be a non-negative float.")
        if seed < 0:
            raise ValueError("Seed must be a non-negative integer.")

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
