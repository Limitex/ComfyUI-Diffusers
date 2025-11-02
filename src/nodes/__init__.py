from ..di import Container
from ._diffusers_pipeline_loader import DiffusersPipelineLoader
from ._diffusers_vae_loader import DiffusersVaeLoader
from .dto import ComfyUIAutoencoderDTO, ComfyUIClipDTO, ComfyUIPipelineDTO

container = Container()
container.wire(modules=[__name__])

NODE_CLASS_MAPPINGS = {
    DiffusersPipelineLoader.__name__: DiffusersPipelineLoader,
    DiffusersVaeLoader.__name__: DiffusersVaeLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    DiffusersPipelineLoader.__name__: "Diffusers Pipeline Loader",
    DiffusersVaeLoader.__name__: "Diffusers VAE Loader",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "ComfyUIPipelineDTO",
    "ComfyUIAutoencoderDTO",
    "ComfyUIClipDTO",
]
