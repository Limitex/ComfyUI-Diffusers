from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Conditioning:
    conditioning: torch.Tensor
    path: str
    prompt: str

    def __post_init__(self) -> None:
        if not isinstance(self.conditioning, torch.Tensor):
            raise ValueError(
                f"Conditioning must be a torch.Tensor, got {type(self.conditioning).__name__}"
            )
        if not isinstance(self.path, str):
            raise ValueError(f"Path must be a string, got {type(self.path).__name__}")
        if not isinstance(self.prompt, str):
            raise ValueError(f"Prompt must be a string, got {type(self.prompt).__name__}")
        if torch.isnan(self.conditioning).any():
            raise ValueError("Conditioning tensor contains NaN values")
        if torch.isinf(self.conditioning).any():
            raise ValueError("Conditioning tensor contains Inf values")
        if not self.path.strip():
            raise ValueError("Path cannot be empty or whitespace only")
        if not self.prompt.strip():
            raise ValueError("Prompt cannot be empty or whitespace only")
