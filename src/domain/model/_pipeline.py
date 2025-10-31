from dataclasses import dataclass

from diffusers import StableDiffusionPipeline


@dataclass
class Pipeline:
    pipeline: StableDiffusionPipeline
    path: str
