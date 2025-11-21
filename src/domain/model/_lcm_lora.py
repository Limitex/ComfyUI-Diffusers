from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LcmLora:
    """LCM LoRA weights for Stream Diffusion."""

    weights: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.weights, dict):
            raise ValueError(f"LCM LoRA weights must be a dict, got {type(self.weights).__name__}")
        if not self.weights:
            raise ValueError("LCM LoRA weights cannot be empty")
