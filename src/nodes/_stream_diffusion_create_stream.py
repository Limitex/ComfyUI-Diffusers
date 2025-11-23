from typing import Any

from dependency_injector.wiring import Provide, inject

from ..di import Container
from ..domain.model import CFGType, CFGTypeEnum, FrameBufferSize, ImageSize, TIndexList
from ..usecase import StreamDiffusionCreateStreamUsecase
from .dto import (
    ComfyUILcmLoraDTO,
    ComfyUIPipelineDTO,
    ComfyUISchedulerDTO,
    ComfyUIStreamDTO,
)

NodeInputMap = dict[str, dict[str, tuple[str, ...] | tuple[str | list[str], dict[str, Any]]]]


class StreamDiffusionCreateStream:
    """Node to create a Stream Diffusion stream."""

    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(
        cls,
    ) -> NodeInputMap:
        return {
            "required": {
                "pipeline": (ComfyUIPipelineDTO.COMFY_TYPE,),
                "scheduler": (ComfyUISchedulerDTO.COMFY_TYPE,),
                "t_index_list": ("LIST",),
                "width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "do_add_noise": ("BOOLEAN", {"default": True}),
                "use_denoising_batch": ("BOOLEAN", {"default": True}),
                "frame_buffer_size": ("INT", {"default": 1, "min": 1, "max": 10000}),
                "cfg_type": (["none", "full", "self", "initialize"], {"default": "none"}),
                "lcm_lora": (ComfyUILcmLoraDTO.COMFY_TYPE,),
                "tiny_vae": ("STRING", {"default": "madebyollin/taesd"}),
                "enable_xformers": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = (ComfyUIStreamDTO.COMFY_TYPE,)
    FUNCTION = "execute"
    CATEGORY = "Diffusers/StreamDiffusion"

    @inject
    def execute(
        self,
        pipeline: ComfyUIPipelineDTO,
        scheduler: ComfyUISchedulerDTO,
        t_index_list: list[int],
        width: int,
        height: int,
        do_add_noise: bool,
        use_denoising_batch: bool,
        frame_buffer_size: int,
        cfg_type: str,
        lcm_lora: ComfyUILcmLoraDTO,
        tiny_vae: str,
        enable_xformers: bool,
        usecase: StreamDiffusionCreateStreamUsecase = Provide[
            Container.stream_diffusion_create_stream_usecase
        ],
    ) -> tuple[ComfyUIStreamDTO]:
        """Create a Stream Diffusion stream.

        Args:
            pipeline: Pipeline DTO
            scheduler: Scheduler DTO
            t_index_list: List of timestep indices
            width: Image width
            height: Image height
            do_add_noise: Whether to add noise
            use_denoising_batch: Whether to use denoising batch
            frame_buffer_size: Size of frame buffer
            cfg_type: CFG type string
            lcm_lora: LCM LoRA DTO
            tiny_vae: Name of tiny VAE model
            enable_xformers: Whether to enable xformers
            usecase: Injected StreamDiffusionCreateStreamUsecase

        Returns:
            Tuple containing ComfyUIStreamDTO
        """
        pipeline_domain = ComfyUIPipelineDTO.to_domain(pipeline)
        scheduler_domain = ComfyUISchedulerDTO.to_domain(scheduler)
        lcm_lora_domain = ComfyUILcmLoraDTO.to_domain(lcm_lora)

        t_index_list_vo = TIndexList(indices=t_index_list)
        image_size = ImageSize(width=width, height=height)
        frame_buffer_size_vo = FrameBufferSize(value=frame_buffer_size)
        cfg_type_vo = CFGType(value=CFGTypeEnum(cfg_type))

        stream = usecase.execute(
            pipeline=pipeline_domain,
            scheduler=scheduler_domain,
            t_index_list=t_index_list_vo,
            image_size=image_size,
            do_add_noise=do_add_noise,
            use_denoising_batch=use_denoising_batch,
            frame_buffer_size=frame_buffer_size_vo,
            cfg_type=cfg_type_vo,
            lcm_lora=lcm_lora_domain,
            tiny_vae_name=tiny_vae,
            enable_xformers=enable_xformers,
        )

        dto = ComfyUIStreamDTO.from_domain(stream)
        return (dto,)
