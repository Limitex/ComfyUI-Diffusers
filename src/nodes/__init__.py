from ..di import Container
from ._diffusers_clip_text_encode import DiffusersClipTextEncode
from ._diffusers_pipeline_loader import DiffusersPipelineLoader
from ._diffusers_sampler import DiffusersSampler
from ._diffusers_scheduler_loader import DiffusersSchedulerLoader
from ._diffusers_vae_loader import DiffusersVaeLoader
from ._lcm_lora_loader import LcmLoraLoader
from ._stream_diffusion_create_stream import StreamDiffusionCreateStream
from ._stream_diffusion_fast_sampler import StreamDiffusionFastSampler
from ._stream_diffusion_sampler import StreamDiffusionSampler
from ._stream_diffusion_warmup import StreamDiffusionWarmup
from .dto import (
    ComfyUIAutoencoderDTO,
    ComfyUIClipDTO,
    ComfyUIConditioningDTO,
    ComfyUILcmLoraDTO,
    ComfyUIPipelineDTO,
    ComfyUISchedulerDTO,
    ComfyUIStreamDTO,
    ComfyUIWarmupStreamDTO,
)

container = Container()
container.wire(modules=[__name__])

NODE_CLASS_MAPPINGS = {
    DiffusersPipelineLoader.__name__: DiffusersPipelineLoader,
    DiffusersVaeLoader.__name__: DiffusersVaeLoader,
    DiffusersClipTextEncode.__name__: DiffusersClipTextEncode,
    DiffusersSampler.__name__: DiffusersSampler,
    DiffusersSchedulerLoader.__name__: DiffusersSchedulerLoader,
    LcmLoraLoader.__name__: LcmLoraLoader,
    StreamDiffusionCreateStream.__name__: StreamDiffusionCreateStream,
    StreamDiffusionWarmup.__name__: StreamDiffusionWarmup,
    StreamDiffusionSampler.__name__: StreamDiffusionSampler,
    StreamDiffusionFastSampler.__name__: StreamDiffusionFastSampler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    DiffusersPipelineLoader.__name__: "Diffusers Pipeline Loader",
    DiffusersVaeLoader.__name__: "Diffusers VAE Loader",
    DiffusersClipTextEncode.__name__: "Diffusers CLIP Text Encode",
    DiffusersSampler.__name__: "Diffusers Sampler",
    DiffusersSchedulerLoader.__name__: "Diffusers Scheduler Loader",
    LcmLoraLoader.__name__: "LCM LoRA Loader",
    StreamDiffusionCreateStream.__name__: "StreamDiffusion Create Stream",
    StreamDiffusionWarmup.__name__: "StreamDiffusion Warmup",
    StreamDiffusionSampler.__name__: "StreamDiffusion Sampler",
    StreamDiffusionFastSampler.__name__: "StreamDiffusion Fast Sampler",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "ComfyUIPipelineDTO",
    "ComfyUIAutoencoderDTO",
    "ComfyUIClipDTO",
    "ComfyUIConditioningDTO",
    "ComfyUISchedulerDTO",
    "ComfyUILcmLoraDTO",
    "ComfyUIStreamDTO",
    "ComfyUIWarmupStreamDTO",
]
