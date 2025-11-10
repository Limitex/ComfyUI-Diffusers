from dataclasses import dataclass


@dataclass(frozen=True)
class Steps:
    value: int

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, int):
            raise ValueError(f"Steps must be an int, got {type(self.value).__name__}")
        if self.value <= 0:
            raise ValueError(f"Steps must be positive, got {self.value}")
