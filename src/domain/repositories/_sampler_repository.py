from abc import ABC, abstractmethod

import torch
from diffusers import AutoencoderKL, StableDiffusionPipeline
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from PIL import Image


class SamplerRepository(ABC):
    @abstractmethod
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
        pass
