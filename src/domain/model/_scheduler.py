from dataclasses import dataclass
from enum import Enum

from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    UniPCMultistepScheduler,
)
from diffusers.schedulers.scheduling_utils import SchedulerMixin


@dataclass(frozen=True)
class Scheduler:
    class Type(Enum):
        DDIM = DDIMScheduler
        DDPM = DDPMScheduler
        DEISMultistep = DEISMultistepScheduler
        DPMSolverMultistep = DPMSolverMultistepScheduler
        DPMSolverSinglestep = DPMSolverSinglestepScheduler
        EulerAncestralDiscrete = EulerAncestralDiscreteScheduler
        EulerDiscrete = EulerDiscreteScheduler
        HeunDiscrete = HeunDiscreteScheduler
        KDPM2AncestralDiscrete = KDPM2AncestralDiscreteScheduler
        KDPM2Discrete = KDPM2DiscreteScheduler
        UniPCMultistep = UniPCMultistepScheduler

    scheduler: SchedulerMixin
    path: str

    def __post_init__(self) -> None:
        if not isinstance(self.scheduler, SchedulerMixin):
            raise ValueError(
                f"Scheduler must inherit from SchedulerMixin, got {type(self.scheduler).__name__}"
            )
        if not isinstance(self.path, str):
            raise ValueError(f"Path must be a string, got {type(self.path).__name__}")
        if not self.path.strip():
            raise ValueError("Path cannot be empty or whitespace only")
