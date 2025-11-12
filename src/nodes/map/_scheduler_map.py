from typing import Final

from ...domain.model import Scheduler


class SchedulerMap:
    """Maps between domain Scheduler.Type enum and ComfyUI string representation.

    This map provides bidirectional conversion between the domain layer's
    type-safe enum and ComfyUI's string-based interface requirements.
    """

    SCHEDULERS: Final[list[str]] = [
        "DDIM",
        "DDPM",
        "DEISMultistep",
        "DPMSolverMultistep",
        "DPMSolverSinglestep",
        "EulerAncestralDiscrete",
        "EulerDiscrete",
        "HeunDiscrete",
        "KDPM2AncestralDiscrete",
        "KDPM2Discrete",
        "UniPCMultistep",
    ]

    _DOMAIN_TO_STRING: dict[Scheduler.Type, str] = {
        Scheduler.Type.DDIM: "DDIM",
        Scheduler.Type.DDPM: "DDPM",
        Scheduler.Type.DEISMultistep: "DEISMultistep",
        Scheduler.Type.DPMSolverMultistep: "DPMSolverMultistep",
        Scheduler.Type.DPMSolverSinglestep: "DPMSolverSinglestep",
        Scheduler.Type.EulerAncestralDiscrete: "EulerAncestralDiscrete",
        Scheduler.Type.EulerDiscrete: "EulerDiscrete",
        Scheduler.Type.HeunDiscrete: "HeunDiscrete",
        Scheduler.Type.KDPM2AncestralDiscrete: "KDPM2AncestralDiscrete",
        Scheduler.Type.KDPM2Discrete: "KDPM2Discrete",
        Scheduler.Type.UniPCMultistep: "UniPCMultistep",
    }

    _STRING_TO_DOMAIN: dict[str, Scheduler.Type] = {v: k for k, v in _DOMAIN_TO_STRING.items()}

    @classmethod
    def from_domain(cls, scheduler_type: Scheduler.Type) -> str:
        """Convert domain Scheduler.Type enum to ComfyUI string representation.

        Args:
            scheduler_type: Scheduler.Type enum value from domain layer

        Returns:
            String representation for ComfyUI interface

        Raises:
            KeyError: If the scheduler_type is not in the mapping
        """
        return cls._DOMAIN_TO_STRING[scheduler_type]

    @classmethod
    def to_domain(cls, scheduler_name: str) -> Scheduler.Type:
        """Convert ComfyUI string to domain Scheduler.Type enum.

        Args:
            scheduler_name: String representation from ComfyUI interface

        Returns:
            Corresponding Scheduler.Type enum value from domain layer

        Raises:
            ValueError: If the scheduler_name is not recognized
        """
        if scheduler_name not in cls._STRING_TO_DOMAIN:
            raise ValueError(
                f"Unknown scheduler type: '{scheduler_name}'. "
                f"Available types: {', '.join(cls.SCHEDULERS)}"
            )
        return cls._STRING_TO_DOMAIN[scheduler_name]
