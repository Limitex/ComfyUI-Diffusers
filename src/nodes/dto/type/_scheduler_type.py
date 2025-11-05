from ....domain.model import Scheduler


class ComfyUISchedulerType:
    SCHEDULERS: list[str] = [
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

    _DOMAIN_TO_STRING_MAP: dict[Scheduler.Type, str] = {
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

    _STRING_TO_DOMAIN_MAP: dict[str, Scheduler.Type] = {
        "DDIM": Scheduler.Type.DDIM,
        "DDPM": Scheduler.Type.DDPM,
        "DEISMultistep": Scheduler.Type.DEISMultistep,
        "DPMSolverMultistep": Scheduler.Type.DPMSolverMultistep,
        "DPMSolverSinglestep": Scheduler.Type.DPMSolverSinglestep,
        "EulerAncestralDiscrete": Scheduler.Type.EulerAncestralDiscrete,
        "EulerDiscrete": Scheduler.Type.EulerDiscrete,
        "HeunDiscrete": Scheduler.Type.HeunDiscrete,
        "KDPM2AncestralDiscrete": Scheduler.Type.KDPM2AncestralDiscrete,
        "KDPM2Discrete": Scheduler.Type.KDPM2Discrete,
        "UniPCMultistep": Scheduler.Type.UniPCMultistep,
    }

    @classmethod
    def from_domain(cls, scheduler: Scheduler.Type) -> str:
        """Convert Scheduler.Type enum to string representation.

        Args:
            scheduler: Scheduler.Type enum value

        Returns:
            String representation of the scheduler type
        """
        return cls._DOMAIN_TO_STRING_MAP[scheduler]

    @classmethod
    def to_domain(cls, scheduler_type: str) -> Scheduler.Type:
        """Convert string type to Scheduler.Type enum.

        Args:
            scheduler_type: String representation of scheduler type

        Returns:
            Corresponding Scheduler.Type enum value

        Raises:
            ValueError: If the scheduler_type is not recognized
        """
        if scheduler_type not in cls._STRING_TO_DOMAIN_MAP:
            msg = f"Unknown scheduler type: {scheduler_type}"
            raise ValueError(msg)
        return cls._STRING_TO_DOMAIN_MAP[scheduler_type]
