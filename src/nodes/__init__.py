from ..di import Container
from .diffusers_pipeline_loader import DiffusersPipelineLoader
from .dto import ComfyUIPipelineDTO

container = Container()
container.wire(modules=[__name__])

NODE_CLASS_MAPPINGS = {DiffusersPipelineLoader.__name__: DiffusersPipelineLoader}
NODE_DISPLAY_NAME_MAPPINGS = {DiffusersPipelineLoader.__name__: "Diffusers Pipeline Loader"}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "ComfyUIPipelineDTO",
]
