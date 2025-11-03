from ..di import Container
from ._diffusers_clip_text_encode import DiffusersClipTextEncode
from ._diffusers_pipeline_loader import DiffusersPipelineLoader
from ._diffusers_sampler import DiffusersSampler
from ._diffusers_vae_loader import DiffusersVaeLoader
from .dto import ComfyUIAutoencoderDTO, ComfyUIClipDTO, ComfyUIConditioningDTO, ComfyUIPipelineDTO

container = Container()
container.wire(modules=[__name__])

NODE_CLASS_MAPPINGS = {
    DiffusersPipelineLoader.__name__: DiffusersPipelineLoader,
    DiffusersVaeLoader.__name__: DiffusersVaeLoader,
    DiffusersClipTextEncode.__name__: DiffusersClipTextEncode,
    DiffusersSampler.__name__: DiffusersSampler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    DiffusersPipelineLoader.__name__: "Diffusers Pipeline Loader",
    DiffusersVaeLoader.__name__: "Diffusers VAE Loader",
    DiffusersClipTextEncode.__name__: "Diffusers CLIP Text Encode",
    DiffusersSampler.__name__: "Diffusers Sampler",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "ComfyUIPipelineDTO",
    "ComfyUIAutoencoderDTO",
    "ComfyUIClipDTO",
    "ComfyUIConditioningDTO",
]
