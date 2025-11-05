from ..domain.model import Autoencoder, Conditioning, Image, Pipeline, Scheduler
from ..service import SamplingService


class SamplerHandler:
    def __init__(self, sampling_service: SamplingService) -> None:
        self.sampling_service = sampling_service

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
        domain_images = self.sampling_service.sample(
            pipeline,
            vae,
            scheduler,
            positive_embeds,
            negative_embeds,
            width,
            height,
            steps,
            cfg,
            seed,
        )
        if domain_images is None:
            raise RuntimeError("Failed to sample images from pipeline.")
        return domain_images
