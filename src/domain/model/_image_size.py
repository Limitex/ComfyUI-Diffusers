from dataclasses import dataclass


@dataclass(frozen=True)
class ImageSize:
    width: int
    height: int

    def __post_init__(self) -> None:
        if isinstance(self.width, bool) or not isinstance(self.width, int):
            raise ValueError(f"Width must be an int, got {type(self.width).__name__}")
        if isinstance(self.height, bool) or not isinstance(self.height, int):
            raise ValueError(f"Height must be an int, got {type(self.height).__name__}")
        if self.width <= 0:
            raise ValueError(f"Width must be positive, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"Height must be positive, got {self.height}")
        if self.width % 8 != 0:
            raise ValueError(f"Width must be divisible by 8, got {self.width}")
        if self.height % 8 != 0:
            raise ValueError(f"Height must be divisible by 8, got {self.height}")
