from abc import ABC, abstractmethod

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


class SamplingRepository(ABC):
    @abstractmethod
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
    ) -> list[Image.Image]:
        pass
