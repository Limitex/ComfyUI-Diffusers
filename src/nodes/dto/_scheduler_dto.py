from dataclasses import dataclass
from typing import ClassVar

from diffusers.schedulers.scheduling_utils import SchedulerMixin

from ...domain.model import Scheduler


@dataclass
class ComfyUISchedulerDTO:
    """Data Transfer Object for Diffusers Scheduler in ComfyUI.

    This DTO is designed to be used across different custom nodes.
    Other node developers can import and use this type for scheduler operations.

    Attributes:
        scheduler: SchedulerMixin instance from diffusers library
        path: Path to the source model directory

    Class Attributes:
        COMFY_TYPE: Type name used in ComfyUI's RETURN_TYPES and INPUT_TYPES

    Example:
        ```python
        from your_node_package.nodes.dto import ComfyUISchedulerDTO

        class YourCustomNode:
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"scheduler": (ComfyUISchedulerDTO.COMFY_TYPE,)}}

            def execute(self, scheduler: ComfyUISchedulerDTO):
                # Use the scheduler for sampling operations
                pass
        ```
    """

    # ClassVar to avoid being treated as a dataclass field
    COMFY_TYPE: ClassVar[str] = "DIFFUSERS_SCHEDULER"

    scheduler: SchedulerMixin
    path: str

    @classmethod
    def from_domain(cls, scheduler: Scheduler) -> "ComfyUISchedulerDTO":
        """Create DTO from domain model.

        Args:
            scheduler: Scheduler from domain layer

        Returns:
            ComfyUISchedulerDTO instance
        """
        return cls(scheduler=scheduler.scheduler, path=scheduler.path)

    @classmethod
    def to_domain(cls, dto: "ComfyUISchedulerDTO") -> Scheduler:
        """Convert DTO back to domain model.

        Args:
            dto: ComfyUISchedulerDTO instance

        Returns:
            Scheduler domain model
        """
        return Scheduler(scheduler=dto.scheduler, path=dto.path)
