from typing import Any

from dependency_injector.wiring import Provide, inject

from ..di import Container
from ..domain.model import NumSamples
from ..usecase import StreamDiffusionFastSampleUsecase
from .dto import ComfyUIImage, ComfyUIImageDTO, ComfyUIWarmupStreamDTO

NodeInputMap = dict[str, dict[str, tuple[str, ...] | tuple[str | list[str], dict[str, Any]]]]


class StreamDiffusionFastSampler:
    """Node for fast sampling from a warmed-up Stream Diffusion stream."""

    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(
        cls,
    ) -> NodeInputMap:
        return {
            "required": {
                "warmup_stream": (ComfyUIWarmupStreamDTO.COMFY_TYPE,),
                "positive_prompt": ("STRING", {"multiline": True}),
                "num": ("INT", {"default": 1, "min": 1, "max": 10000}),
            },
        }

    RETURN_TYPES = (ComfyUIImage.COMFY_TYPE,)
    FUNCTION = "execute"
    CATEGORY = "Diffusers/StreamDiffusion"

    @inject
    def execute(
        self,
        warmup_stream: ComfyUIWarmupStreamDTO,
        positive_prompt: str,
        num: int,
        usecase: StreamDiffusionFastSampleUsecase = Provide[
            Container.stream_diffusion_fast_sample_usecase
        ],
    ) -> tuple[ComfyUIImageDTO]:
        """Fast sample images from a warmed-up stream.

        Args:
            warmup_stream: Warmed-up stream DTO
            positive_prompt: Prompt text
            num: Number of images to generate
            usecase: Injected StreamDiffusionFastSampleUsecase

        Returns:
            Tuple containing image tensor
        """
        stream_domain = ComfyUIWarmupStreamDTO.to_domain(warmup_stream)
        num_vo = NumSamples(value=num)

        images = usecase.execute(
            stream=stream_domain,
            prompt=positive_prompt,
            num_samples=num_vo,
        )

        result_dto = ComfyUIImage.from_domains(images)
        return (result_dto,)
