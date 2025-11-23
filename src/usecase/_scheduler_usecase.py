import torch
from diffusers.schedulers.scheduling_utils import SchedulerMixin

from ..domain.model import Pipeline, Scheduler
from ..domain.repositories import SchedulerRepository


class SchedulerUsecase:
    def __init__(self, scheduler_repo: SchedulerRepository) -> None:
        self.scheduler_repo = scheduler_repo
        self.dtype = torch.float16

    def execute(self, pipeline: Pipeline, scheduler_type: Scheduler.Type) -> Scheduler:
        scheduler_obj: SchedulerMixin = self.scheduler_repo.create_scheduler(
            pipeline.path,
            self.dtype,
            scheduler_type,
        )

        return Scheduler(scheduler=scheduler_obj, path=pipeline.path)
