from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LcmLora:
    """LCM LoRA weights for Stream Diffusion."""

    weights: dict[str, torch.Tensor]

    def __post_init__(self) -> None:
        if not isinstance(self.weights, dict):
            raise ValueError(f"LCM LoRA weights must be a dict, got {type(self.weights).__name__}")
        if not self.weights:
            raise ValueError("LCM LoRA weights cannot be empty")
