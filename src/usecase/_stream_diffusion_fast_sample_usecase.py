from ..domain.model import Image, NumSamples, StreamDiffusionStream
from ..domain.repositories import StreamDiffusionRepository


class StreamDiffusionFastSampleUsecase:
    def __init__(self, stream_diffusion_repo: StreamDiffusionRepository) -> None:
        self.stream_diffusion_repo = stream_diffusion_repo

    def execute(
        self,
        stream: StreamDiffusionStream,
        prompt: str,
        num_samples: NumSamples,
    ) -> list[Image]:
        """Fast sample images from a warmed-up Stream Diffusion stream.

        Args:
            stream: Warmed-up stream to sample from
            prompt: Prompt text
            num_samples: Number of images to generate

        Returns:
            List of generated images as domain models

        Raises:
            RuntimeError: If sampling fails
        """
        # Update prompt
        self.stream_diffusion_repo.update_prompt(stream=stream.stream, prompt=prompt)

        # Sample images
        images = self.stream_diffusion_repo.sample_txt2img(
            stream=stream.stream,
            num_samples=num_samples,
        )

        if not images:
            raise RuntimeError("Stream Diffusion repository returned no images")

        return [Image(image=img) for img in images]
