import folder_paths  # pyright: ignore[reportMissingImports]
from dependency_injector.wiring import Provide, inject

from ..di import Container
from ..domain.model import PipelineModel
from ..ui.pipeline_handler import PipelineHandler


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

    RETURN_TYPES = ("PIPELINE",)
    FUNCTION = "execute"
    CATEGORY = "Diffusers"

    @inject
    def execute(
        self,
        checkpoint_name: str,
        handler: PipelineHandler = Provide[Container.pipeline_handler_provider],
    ) -> tuple[PipelineModel]:
        pipeline_model = handler.create(checkpoint_name)
        return (pipeline_model,)
