from dependency_injector.wiring import Provide, inject

from ..di import Container
from ..usecase import SamplerUsecase
from .dto import (
    ComfyUIAutoencoderDTO,
    ComfyUIConditioningDTO,
    ComfyUIImage,
    ComfyUIImageDTO,
    ComfyUIPipelineDTO,
    ComfyUISchedulerDTO,
)


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
                "vae": (ComfyUIAutoencoderDTO.COMFY_TYPE,),
                "scheduler": (ComfyUISchedulerDTO.COMFY_TYPE,),
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
        vae: ComfyUIAutoencoderDTO,
        scheduler: ComfyUISchedulerDTO,
        positive_embeds: ComfyUIConditioningDTO,
        negative_embeds: ComfyUIConditioningDTO,
        width: int,
        height: int,
        steps: int,
        cfg: float,
        seed: int,
        usecase: SamplerUsecase = Provide[Container.sampler_usecase],
    ) -> tuple[ComfyUIImageDTO]:
        pipeline_domain = ComfyUIPipelineDTO.to_domain(pipeline)
        vae_domain = ComfyUIAutoencoderDTO.to_domain(vae)
        scheduler_domain = ComfyUISchedulerDTO.to_domain(scheduler)
        positive_embeds_domain = ComfyUIConditioningDTO.to_domain(positive_embeds)
        negative_embeds_domain = ComfyUIConditioningDTO.to_domain(negative_embeds)
        images_model = usecase.execute(
            pipeline_domain,
            vae_domain,
            scheduler_domain,
            positive_embeds_domain,
            negative_embeds_domain,
            width,
            height,
            steps,
            cfg,
            seed,
        )
        sampler_dto = ComfyUIImage.from_domains(images_model)
        return (sampler_dto,)
