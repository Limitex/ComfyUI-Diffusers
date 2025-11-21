from dataclasses import dataclass


@dataclass(frozen=True)
class TIndexList:
    """List of timestep indices for Stream Diffusion."""

    indices: list[int]

    def __post_init__(self) -> None:
        if not isinstance(self.indices, list):
            raise ValueError(f"Indices must be a list, got {type(self.indices).__name__}")
        if not self.indices:
            raise ValueError("Indices list cannot be empty")
        for idx, value in enumerate(self.indices):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"Index at position {idx} must be an int, got {type(value).__name__}"
                )
