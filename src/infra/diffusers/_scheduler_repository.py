import folder_paths  # pyright: ignore[reportMissingImports]
import torch
from diffusers.schedulers.scheduling_utils import SchedulerMixin

from ...domain.model import Scheduler
from ...domain.repositories import SchedulerRepository


class DiffusersSchedulerRepository(SchedulerRepository):
    def __init__(self) -> None:
        self.cache_dir = folder_paths.get_temp_directory()

    def create_scheduler(
        self, model_path: str, dtype: torch.dtype, scheduler_type: Scheduler.Type
    ) -> SchedulerMixin:
        scheduler: SchedulerMixin = scheduler_type.value.from_pretrained(
            pretrained_model_name_or_path=model_path,
            subfolder="scheduler",
            torch_dtype=dtype,
            cache_dir=self.cache_dir,
        )
        return scheduler
