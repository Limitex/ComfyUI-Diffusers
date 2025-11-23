from abc import ABC, abstractmethod

import torch
from diffusers import StableDiffusionPipeline
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from PIL import Image
from streamdiffusion import StreamDiffusion  # type: ignore[import-untyped]

from ..model import (
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


class StreamDiffusionRepository(ABC):
    @abstractmethod
    def create_stream(
        self,
        pipeline: StableDiffusionPipeline,
        scheduler: SchedulerMixin,
        t_index_list: TIndexList,
        image_size: ImageSize,
        do_add_noise: bool,
        use_denoising_batch: bool,
        frame_buffer_size: FrameBufferSize,
        cfg_type: CFGType,
        lcm_lora_weights: dict[str, torch.Tensor],
        tiny_vae_name: str,
        enable_xformers: bool,
    ) -> StreamDiffusion:
        """Create a Stream Diffusion stream instance.

        Args:
            pipeline: StableDiffusionPipeline instance
            scheduler: Scheduler to attach to the pipeline
            t_index_list: List of timestep indices
            image_size: Image dimensions
            do_add_noise: Whether to add noise
            use_denoising_batch: Whether to use denoising batch
            frame_buffer_size: Size of frame buffer
            cfg_type: CFG type configuration
            lcm_lora_weights: LCM LoRA weights
            tiny_vae_name: Name of tiny VAE model
            enable_xformers: Whether to enable xformers memory efficient attention

        Returns:
            StreamDiffusion instance
        """

    @abstractmethod
    def warmup_stream(
        self,
        stream: StreamDiffusion,
        warmup_count: WarmupCount,
        input_image: Image.Image | None = None,
    ) -> None:
        """Warm up the stream with given parameters.

        Args:
            stream: StreamDiffusion instance
            warmup_count: Number of warmup iterations
            input_image: Optional input image for img2img warmup
        """

    @abstractmethod
    def prepare_stream(
        self,
        stream: StreamDiffusion,
        prompt: str,
        negative_prompt: str,
        steps: Steps,
        cfg: CFGScale,
        delta: Delta,
        seed: Seed,
    ) -> None:
        """Prepare the stream with given parameters.

        Args:
            stream: StreamDiffusion instance
            prompt: Prompt text
            negative_prompt: Negative prompt text
            steps: Number of inference steps
            cfg: CFG scale value
            delta: Delta value
            seed: Random seed
        """

    @abstractmethod
    def update_prompt(self, stream: StreamDiffusion, prompt: str) -> None:
        """Update the prompt for the stream.

        Args:
            stream: StreamDiffusion instance
            prompt: New prompt text
        """

    @abstractmethod
    def sample_txt2img(
        self,
        stream: StreamDiffusion,
        num_samples: NumSamples,
    ) -> list[Image.Image]:
        """Generate images using txt2img.

        Args:
            stream: StreamDiffusion instance
            num_samples: Number of images to generate

        Returns:
            List of generated PIL images
        """

    @abstractmethod
    def sample_with_images(
        self,
        stream: StreamDiffusion,
        num_samples: NumSamples,
        input_images: list[Image.Image] | None,
    ) -> list[Image.Image]:
        """Generate images with optional input images.

        Args:
            stream: StreamDiffusion instance
            num_samples: Number of images to generate
            input_images: Optional list of input images

        Returns:
            List of generated PIL images
        """
