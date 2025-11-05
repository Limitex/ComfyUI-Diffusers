from ..domain.model import Pipeline, Scheduler
from ..service import SchedulerService


class SchedulerHandler:
    def __init__(self, scheduler_service: SchedulerService) -> None:
        self.scheduler_service = scheduler_service

    def create(self, pipeline: Pipeline, scheduler_type: Scheduler.Type) -> Scheduler:
        if pipeline is None or pipeline.pipeline is None:
            raise ValueError("Pipeline is None.")
        return self.scheduler_service.create_scheduler(pipeline, scheduler_type)
