from dataclasses import dataclass

from diffusers import StableDiffusionPipeline


@dataclass(frozen=True)
class Pipeline:
    pipeline: StableDiffusionPipeline
    path: str

    def __post_init__(self) -> None:
        if not isinstance(self.pipeline, StableDiffusionPipeline):
            raise ValueError(
                f"Pipeline must be an instance of StableDiffusionPipeline, got {type(self.pipeline).__name__}"
            )
        if not isinstance(self.path, str):
            raise ValueError(f"Path must be a string, got {type(self.path).__name__}")
        if not self.path.strip():
            raise ValueError("Path cannot be empty or whitespace only")
