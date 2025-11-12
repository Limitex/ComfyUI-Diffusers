from dependency_injector.wiring import Provide, inject

from ..di import Container
from ..usecase import SchedulerUsecase
from .dto import ComfyUIPipelineDTO, ComfyUISchedulerDTO
from .map import SchedulerMap


class DiffusersSchedulerLoader:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[str] | tuple[list[str]]]]:
        return {
            "required": {
                "pipeline": (ComfyUIPipelineDTO.COMFY_TYPE,),
                "scheduler_name": (SchedulerMap.SCHEDULERS,),
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
        usecase: SchedulerUsecase = Provide[Container.scheduler_usecase],
    ) -> tuple[ComfyUISchedulerDTO]:
        domain = ComfyUIPipelineDTO.to_domain(pipeline)
        scheduler_type = SchedulerMap.to_domain(scheduler_name)
        scheduler_domain = usecase.execute(domain, scheduler_type)
        scheduler_dto = ComfyUISchedulerDTO.from_domain(scheduler_domain)
        return (scheduler_dto,)
