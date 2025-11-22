import copy
from typing import Any

import torch
from comfy.model_management import get_torch_device  # pyright: ignore[reportMissingImports]
from diffusers import AutoencoderTiny, StableDiffusionPipeline
from PIL import Image
from streamdiffusion import StreamDiffusion  # type: ignore[import-untyped]
from streamdiffusion.image_utils import postprocess_image  # type: ignore[import-untyped]

from ...domain.model import (
    CFGScale,
    CFGType,
    Delta,
    FrameBufferSize,
    ImageSize,
    NumSamples,
    Seed,
    Steps,
    TIndexList,
    WarmupCount,
)
from ...domain.repositories import StreamDiffusionRepository
from ._cache import get_cache_path


class DiffusersStreamDiffusionRepository(StreamDiffusionRepository):
    def __init__(self) -> None:
        self.device = get_torch_device()
        self.cache_dir = get_cache_path()
        self.dtype = torch.float16

    def create_stream(
        self,
        pipeline: StableDiffusionPipeline,
        t_index_list: TIndexList,
        image_size: ImageSize,
        do_add_noise: bool,
        use_denoising_batch: bool,
        frame_buffer_size: FrameBufferSize,
        cfg_type: CFGType,
        lcm_lora_weights: dict[str, Any],
        tiny_vae_name: str,
        enable_xformers: bool,
    ) -> Any:
        """Create a Stream Diffusion stream instance."""
        # Deep copy to avoid modifying the original pipeline
        # Note: load_lcm_lora() and fuse_lora() modify the pipeline,
        # so we need to copy it to allow reusing the original pipeline in ComfyUI
        pipeline_copy = copy.deepcopy(pipeline)
        lora_weights_copy = copy.deepcopy(lcm_lora_weights)

        # Create stream
        stream = StreamDiffusion(
            pipe=pipeline_copy,
            t_index_list=t_index_list.indices,
            torch_dtype=self.dtype,
            width=image_size.width,
            height=image_size.height,
            do_add_noise=do_add_noise,
            use_denoising_batch=use_denoising_batch,
            frame_buffer_size=frame_buffer_size.value,
            cfg_type=cfg_type.value.value,  # Get the string value from enum
        )

        # Load and fuse LCM LoRA
        stream.load_lcm_lora(lora_weights_copy)
        stream.fuse_lora()

        # Load tiny VAE
        stream.vae = AutoencoderTiny.from_pretrained(  # type: ignore[no-untyped-call]
            pretrained_model_name_or_path=tiny_vae_name,
            torch_dtype=self.dtype,
            cache_dir=self.cache_dir,
        ).to(device=pipeline_copy.device, dtype=pipeline_copy.dtype)  # type: ignore[attr-defined]

        # Enable xformers if requested
        if enable_xformers:
            pipeline_copy.enable_xformers_memory_efficient_attention()  # type: ignore[attr-defined]

        return stream

    def warmup_stream(
        self,
        stream: Any,
        negative_prompt: str,
        steps: Steps,
        cfg: CFGScale,
        delta: Delta,
        seed: Seed,
        warmup_count: WarmupCount,
    ) -> None:
        """Warm up the stream with given parameters."""
        stream.prepare(
            prompt="",
            negative_prompt=negative_prompt,
            num_inference_steps=steps.value,
            guidance_scale=cfg.value,
            delta=delta.value,
            seed=seed.value,
        )

        for _ in range(warmup_count.value):
            stream()

    def update_prompt(self, stream: Any, prompt: str) -> None:
        """Update the prompt for the stream."""
        stream.update_prompt(prompt)

    def sample_txt2img(
        self,
        stream: Any,
        num_samples: NumSamples,
    ) -> list[Image.Image]:
        """Generate images using txt2img."""
        result: list[Image.Image] = []
        for _ in range(num_samples.value):
            x_output = stream.txt2img()
            image = postprocess_image(x_output, output_type="pil")[0]
            result.append(image)
        return result

    def sample_with_images(
        self,
        stream: Any,
        prompt: str,
        negative_prompt: str,
        steps: Steps,
        cfg: CFGScale,
        delta: Delta,
        seed: Seed,
        num_samples: NumSamples,
        warmup_count: WarmupCount,
        input_images: list[Image.Image] | None,
    ) -> list[Image.Image]:
        """Generate images with optional input images."""
        # Prepare stream
        stream.prepare(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps.value,
            guidance_scale=cfg.value,
            delta=delta.value,
            seed=seed.value,
        )

        # Resize input images if provided
        if input_images is not None:
            resized_images = [img.resize((stream.width, stream.height)) for img in input_images]
        else:
            resized_images = None

        # Warmup
        for _ in range(warmup_count.value):
            stream()

        # Generate images
        result: list[Image.Image] = []
        if resized_images is None:
            # Text-to-image: Generate num_samples images
            for _ in range(num_samples.value):
                x_output = stream.txt2img()
                image = postprocess_image(x_output, output_type="pil")[0]
                result.append(image)
        else:
            # Image-to-image: Process each input image num_samples times
            for _ in range(num_samples.value):
                for img in resized_images:
                    x_output = stream(img)
                    image = postprocess_image(x_output, output_type="pil")[0]
                    result.append(image)

        return result
