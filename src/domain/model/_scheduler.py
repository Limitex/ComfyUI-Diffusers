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


@dataclass
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
