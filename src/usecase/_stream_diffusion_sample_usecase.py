from ..domain.model import (
    CFGScale,
    Delta,
    Image,
    NumSamples,
    Seed,
    Steps,
    StreamDiffusionStream,
    WarmupCount,
)
from ..domain.repositories import StreamDiffusionRepository


class StreamDiffusionSampleUsecase:
    def __init__(self, stream_diffusion_repo: StreamDiffusionRepository) -> None:
        self.stream_diffusion_repo = stream_diffusion_repo

    def execute(
        self,
        stream: StreamDiffusionStream,
        prompt: str,
        negative_prompt: str,
        steps: Steps,
        cfg: CFGScale,
        delta: Delta,
        seed: Seed,
        num_samples: NumSamples,
        warmup_count: WarmupCount,
        input_images: list[Image] | None,
    ) -> list[Image]:
        """Sample images from a Stream Diffusion stream.

        Args:
            stream: Stream to sample from
            prompt: Prompt text
            negative_prompt: Negative prompt text
            steps: Number of inference steps
            cfg: CFG scale value
            delta: Delta value
            seed: Random seed
            num_samples: Number of images to generate
            warmup_count: Number of warmup iterations
            input_images: Optional list of input images

        Returns:
            List of generated images as domain models

        Raises:
            RuntimeError: If sampling fails
        """
        # Convert domain images to PIL images if provided
        pil_images = None
        if input_images is not None:
            pil_images = [img.image for img in input_images]

        images = self.stream_diffusion_repo.sample_with_images(
            stream=stream.stream,
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            cfg=cfg,
            delta=delta,
            seed=seed,
            num_samples=num_samples,
            warmup_count=warmup_count,
            input_images=pil_images,
        )

        if not images:
            raise RuntimeError("Stream Diffusion repository returned no images")

        return [Image(image=img) for img in images]
