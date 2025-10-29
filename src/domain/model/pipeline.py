from dataclasses import dataclass

from diffusers import StableDiffusionPipeline


@dataclass
class PipelineModel:
    pipeline: StableDiffusionPipeline
    path: str
