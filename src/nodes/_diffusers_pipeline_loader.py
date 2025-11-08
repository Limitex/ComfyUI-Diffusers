import folder_paths  # pyright: ignore[reportMissingImports]
from dependency_injector.wiring import Provide, inject

from ..di import Container
from ..usecase import PipelineUsecase
from .dto import ComfyUIClipDTO, ComfyUIPipelineDTO


class DiffusersPipelineLoader:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[tuple[str, ...], ...]]]:
        return {
            "required": {
                "checkpoint_name": (folder_paths.get_filename_list("checkpoints"),),
            }
        }

    RETURN_TYPES = (
        ComfyUIPipelineDTO.COMFY_TYPE,
        ComfyUIClipDTO.COMFY_TYPE,
    )
    FUNCTION = "execute"
    CATEGORY = "Diffusers"

    @inject
    def execute(
        self,
        checkpoint_name: str,
        usecase: PipelineUsecase = Provide[Container.pipeline_usecase],
    ) -> tuple[ComfyUIPipelineDTO, ComfyUIClipDTO]:
        checkpoint_path = folder_paths.get_full_path("checkpoints", checkpoint_name)
        pipeline_model, clip_model = usecase.execute(checkpoint_path)
        pipeline_dto = ComfyUIPipelineDTO.from_domain(pipeline_model)
        clip_dto = ComfyUIClipDTO.from_domain(clip_model)
        return (
            pipeline_dto,
            clip_dto,
        )
