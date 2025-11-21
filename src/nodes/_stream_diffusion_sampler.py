from typing import Any

import numpy as np
from dependency_injector.wiring import Provide, inject
from PIL import Image as PilImage

from ..di import Container
from ..domain.model import CFGScale, Delta, Image, NumSamples, Seed, Steps, WarmupCount
from ..usecase import StreamDiffusionSampleUsecase
from .dto import ComfyUIImage, ComfyUIImageDTO, ComfyUIStreamDTO

NodeInputMap = dict[str, dict[str, tuple[str, ...] | tuple[str | list[str], dict[str, Any]]]]


class StreamDiffusionSampler:
    """Node to sample images from a Stream Diffusion stream."""

    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(
        cls,
    ) -> NodeInputMap:
        return {
            "required": {
                "stream": (ComfyUIStreamDTO.COMFY_TYPE,),
                "positive_prompt": ("STRING", {"multiline": True}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 100.0}),
                "delta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "num": ("INT", {"default": 1, "min": 1, "max": 10000}),
                "warmup": ("INT", {"default": 1, "min": 0, "max": 10000}),
            },
            "optional": {
                "image": (ComfyUIImage.COMFY_TYPE,),
            },
        }

    RETURN_TYPES = (ComfyUIImage.COMFY_TYPE,)
    FUNCTION = "execute"
    CATEGORY = "Diffusers/StreamDiffusion"

    @inject
    def execute(
        self,
        stream: ComfyUIStreamDTO,
        positive_prompt: str,
        negative_prompt: str,
        steps: int,
        cfg: float,
        delta: float,
        seed: int,
        num: int,
        warmup: int,
        image: ComfyUIImageDTO | None = None,
        usecase: StreamDiffusionSampleUsecase = Provide[Container.stream_diffusion_sample_usecase],
    ) -> tuple[ComfyUIImageDTO]:
        """Sample images from a Stream Diffusion stream.

        Args:
            stream: Stream DTO
            positive_prompt: Prompt text
            negative_prompt: Negative prompt text
            steps: Number of inference steps
            cfg: CFG scale value
            delta: Delta value
            seed: Random seed
            num: Number of images to generate
            warmup: Number of warmup iterations
            image: Optional input image tensor
            usecase: Injected StreamDiffusionSampleUsecase

        Returns:
            Tuple containing image tensor
        """
        stream_domain = ComfyUIStreamDTO.to_domain(stream)

        steps_vo = Steps(value=steps)
        cfg_vo = CFGScale(value=cfg)
        delta_vo = Delta(value=delta)
        seed_vo = Seed(value=seed)
        num_vo = NumSamples(value=num)
        warmup_vo = WarmupCount(value=warmup)

        # Convert input images if provided
        input_images_domain: list[Image] | None = None
        if image is not None:
            # ComfyUI images are in format [B, H, W, C] with values in [0, 1]
            images_np = image.cpu().numpy()
            input_images_domain = []
            for img_np in images_np:
                # Convert to uint8 and create PIL image
                img_uint8 = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
                pil_img = PilImage.fromarray(img_uint8)
                input_images_domain.append(Image(image=pil_img))

        images = usecase.execute(
            stream=stream_domain,
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            steps=steps_vo,
            cfg=cfg_vo,
            delta=delta_vo,
            seed=seed_vo,
            num_samples=num_vo,
            warmup_count=warmup_vo,
            input_images=input_images_domain,
        )

        result_dto = ComfyUIImage.from_domains(images)
        return (result_dto,)
