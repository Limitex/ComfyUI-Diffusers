from dataclasses import dataclass


@dataclass(frozen=True)
class CFGScale:
    value: float

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, float):
            raise ValueError(f"CFG scale must be a float, got {type(self.value).__name__}")
        if self.value < 0.0:
            raise ValueError(f"CFG scale must be non-negative, got {self.value}")
