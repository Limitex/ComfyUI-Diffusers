import torch

from ..domain.model import Autoencoder, Conditioning, Image, Pipeline, Scheduler
from ..domain.repositories import SamplingRepository


class SamplingService:
    def __init__(self, sampling_repo: SamplingRepository) -> None:
        self.sampling_repo = sampling_repo
        self.dtype = torch.float32

    def sample(
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
        images = self.sampling_repo.sample(
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
            raise RuntimeError("Sampling repository returned no images.")
        images_domain: list[Image] = [Image(image=img) for img in images]
        return images_domain
