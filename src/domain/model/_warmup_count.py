from dataclasses import dataclass


@dataclass(frozen=True)
class WarmupCount:
    """Number of warmup iterations for Stream Diffusion."""

    value: int

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, int):
            raise ValueError(f"Warmup count must be an int, got {type(self.value).__name__}")
        if self.value < 0:
            raise ValueError(f"Warmup count must be non-negative, got {self.value}")
