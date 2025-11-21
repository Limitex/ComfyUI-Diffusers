from typing import Any

from dependency_injector.wiring import Provide, inject

from ..di import Container
from ..domain.model import CFGScale, Delta, Seed, Steps, WarmupCount
from ..usecase import StreamDiffusionWarmupUsecase
from .dto import ComfyUIStreamDTO, ComfyUIWarmupStreamDTO

NodeInputMap = dict[str, dict[str, tuple[str, ...] | tuple[str | list[str], dict[str, Any]]]]


class StreamDiffusionWarmup:
    """Node to warm up a Stream Diffusion stream."""

    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(
        cls,
    ) -> NodeInputMap:
        return {
            "required": {
                "stream": (ComfyUIStreamDTO.COMFY_TYPE,),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 100.0}),
                "delta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "warmup": ("INT", {"default": 1, "min": 0, "max": 10000}),
            },
        }

    RETURN_TYPES = (ComfyUIWarmupStreamDTO.COMFY_TYPE,)
    FUNCTION = "execute"
    CATEGORY = "Diffusers/StreamDiffusion"

    @inject
    def execute(
        self,
        stream: ComfyUIStreamDTO,
        negative_prompt: str,
        steps: int,
        cfg: float,
        delta: float,
        seed: int,
        warmup: int,
        usecase: StreamDiffusionWarmupUsecase = Provide[Container.stream_diffusion_warmup_usecase],
    ) -> tuple[ComfyUIWarmupStreamDTO]:
        """Warm up a Stream Diffusion stream.

        Args:
            stream: Stream DTO
            negative_prompt: Negative prompt text
            steps: Number of inference steps
            cfg: CFG scale value
            delta: Delta value
            seed: Random seed
            warmup: Number of warmup iterations
            usecase: Injected StreamDiffusionWarmupUsecase

        Returns:
            Tuple containing ComfyUIWarmupStreamDTO
        """
        stream_domain = ComfyUIStreamDTO.to_domain(stream)

        steps_vo = Steps(value=steps)
        cfg_vo = CFGScale(value=cfg)
        delta_vo = Delta(value=delta)
        seed_vo = Seed(value=seed)
        warmup_vo = WarmupCount(value=warmup)

        warmed_stream = usecase.execute(
            stream=stream_domain,
            negative_prompt=negative_prompt,
            steps=steps_vo,
            cfg=cfg_vo,
            delta=delta_vo,
            seed=seed_vo,
            warmup_count=warmup_vo,
        )

        dto = ComfyUIWarmupStreamDTO.from_domain(warmed_stream)
        return (dto,)
