from abc import ABC, abstractmethod

import torch
from diffusers.schedulers.scheduling_utils import SchedulerMixin

from ...domain.model import Scheduler


class SchedulerRepository(ABC):
    @abstractmethod
    def create_scheduler(
        self, model_path: str, dtype: torch.dtype, scheduler_type: Scheduler.Type
    ) -> SchedulerMixin:
        pass
