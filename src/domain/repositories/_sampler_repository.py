from abc import ABC, abstractmethod

import torch
from diffusers import AutoencoderKL, StableDiffusionPipeline
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from PIL import Image

from ..model import CFGScale, ImageSize, Seed, Steps


class SamplerRepository(ABC):
    @abstractmethod
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
        pass
