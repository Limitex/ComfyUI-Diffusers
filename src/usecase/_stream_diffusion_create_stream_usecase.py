from ..domain.model import (
    CFGType,
    FrameBufferSize,
    ImageSize,
    LcmLora,
    Pipeline,
    StreamDiffusionStream,
    TIndexList,
)
from ..domain.repositories import StreamDiffusionRepository


class StreamDiffusionCreateStreamUsecase:
    def __init__(self, stream_diffusion_repo: StreamDiffusionRepository) -> None:
        self.stream_diffusion_repo = stream_diffusion_repo

    def execute(
        self,
        pipeline: Pipeline,
        t_index_list: TIndexList,
        image_size: ImageSize,
        do_add_noise: bool,
        use_denoising_batch: bool,
        frame_buffer_size: FrameBufferSize,
        cfg_type: CFGType,
        lcm_lora: LcmLora,
        tiny_vae_name: str,
        enable_xformers: bool,
    ) -> StreamDiffusionStream:
        """Create a Stream Diffusion stream.

        Args:
            pipeline: Pipeline model
            t_index_list: List of timestep indices
            image_size: Image dimensions
            do_add_noise: Whether to add noise
            use_denoising_batch: Whether to use denoising batch
            frame_buffer_size: Size of frame buffer
            cfg_type: CFG type configuration
            lcm_lora: LCM LoRA model
            tiny_vae_name: Name of tiny VAE model
            enable_xformers: Whether to enable xformers

        Returns:
            StreamDiffusionStream domain model

        Raises:
            RuntimeError: If stream creation fails
        """
        stream = self.stream_diffusion_repo.create_stream(
            pipeline=pipeline.pipeline,
            t_index_list=t_index_list,
            image_size=image_size,
            do_add_noise=do_add_noise,
            use_denoising_batch=use_denoising_batch,
            frame_buffer_size=frame_buffer_size,
            cfg_type=cfg_type,
            lcm_lora_weights=lcm_lora.weights,
            tiny_vae_name=tiny_vae_name,
            enable_xformers=enable_xformers,
        )

        if stream is None:
            raise RuntimeError("Failed to create Stream Diffusion stream")

        return StreamDiffusionStream(stream=stream)
