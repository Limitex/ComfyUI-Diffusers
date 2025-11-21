from dataclasses import dataclass


@dataclass(frozen=True)
class Delta:
    """Delta value for Stream Diffusion."""

    value: float

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, float):
            raise ValueError(f"Delta must be a float, got {type(self.value).__name__}")
        if not (0.0 <= self.value <= 1.0):
            raise ValueError(f"Delta must be between 0.0 and 1.0, got {self.value}")
