from dataclasses import dataclass


@dataclass(frozen=True)
class Seed:
    value: int

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, int):
            raise ValueError(f"Seed must be an int, got {type(self.value).__name__}")
