import folder_paths  # pyright: ignore[reportMissingImports]
from dependency_injector.wiring import Provide, inject

from ..di import Container
from ..ui import PipelineHandler
from .dto import ComfyUIPipelineDTO


class DiffusersPipelineLoader:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s) -> dict[str, dict[str, tuple[tuple[str, ...], ...]]]:
        return {
            "required": {
                "checkpoint_name": (folder_paths.get_filename_list("checkpoints"),),
            }
        }

    RETURN_TYPES = (ComfyUIPipelineDTO.COMFY_TYPE,)
    FUNCTION = "execute"
    CATEGORY = "Diffusers"

    @inject
    def execute(
        self,
        checkpoint_name: str,
        handler: PipelineHandler = Provide[Container.pipeline_handler_provider],
    ) -> tuple[ComfyUIPipelineDTO]:
        pipeline_model = handler.create(checkpoint_name)
        pipeline_dto = ComfyUIPipelineDTO.from_domain(pipeline_model)
        return (pipeline_dto,)
