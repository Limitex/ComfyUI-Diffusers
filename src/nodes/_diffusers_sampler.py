from dependency_injector.wiring import Provide, inject

from ..di import Container
from ..ui import SamplerHandler
from .dto import ComfyUIConditioningDTO, ComfyUIImage, ComfyUIImageDTO, ComfyUIPipelineDTO


class DiffusersSampler:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(
        cls,
    ) -> dict[str, dict[str, tuple[str, ...] | tuple[str, dict[str, int | float]]]]:
        return {
            "required": {
                "pipeline": (ComfyUIPipelineDTO.COMFY_TYPE,),
                "positive_embeds": (ComfyUIConditioningDTO.COMFY_TYPE,),
                "negative_embeds": (ComfyUIConditioningDTO.COMFY_TYPE,),
                "width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1.0}),
                "height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1.0}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1.0}),
                "cfg": (
                    "FLOAT",
                    {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1.0}),
            },
        }

    RETURN_TYPES = (ComfyUIImage.COMFY_TYPE,)
    FUNCTION = "execute"
    CATEGORY = "Diffusers"

    @inject
    def execute(
        self,
        pipeline: ComfyUIPipelineDTO,
        positive_embeds: ComfyUIConditioningDTO,
        negative_embeds: ComfyUIConditioningDTO,
        width: int,
        height: int,
        steps: int,
        cfg: float,
        seed: int,
        handler: SamplerHandler = Provide[Container.sampler_handler],
    ) -> tuple[ComfyUIImageDTO]:
        images_model = handler.sample(
            pipeline.pipeline,
            positive_embeds.conditioning,
            negative_embeds.conditioning,
            width,
            height,
            steps,
            cfg,
            seed,
        )
        sampler_dto = ComfyUIImage.from_domains(images_model)
        return (sampler_dto,)
