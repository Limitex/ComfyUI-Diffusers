from dependency_injector.wiring import Provide, inject

from ..di import Container
from ..ui import SchedulerHandler
from .dto import ComfyUIPipelineDTO, ComfyUISchedulerDTO
from .type import ComfyUISchedulerType


class DiffusersSchedulerLoader:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[str] | tuple[list[str]]]]:
        return {
            "required": {
                "pipeline": (ComfyUIPipelineDTO.COMFY_TYPE,),
                "scheduler_name": (list(ComfyUISchedulerType.SCHEDULERS),),
            }
        }

    RETURN_TYPES = (ComfyUISchedulerDTO.COMFY_TYPE,)
    FUNCTION = "execute"
    CATEGORY = "Diffusers"

    @inject
    def execute(
        self,
        pipeline: ComfyUIPipelineDTO,
        scheduler_name: str,
        handler: SchedulerHandler = Provide[Container.scheduler_handler],
    ) -> tuple[ComfyUISchedulerDTO]:
        domain = ComfyUIPipelineDTO.to_domain(pipeline)
        scheduler_type = ComfyUISchedulerType.to_domain(scheduler_name)
        scheduler_domain = handler.create(domain, scheduler_type)
        scheduler_dto = ComfyUISchedulerDTO.from_domain(scheduler_domain)
        return (scheduler_dto,)
