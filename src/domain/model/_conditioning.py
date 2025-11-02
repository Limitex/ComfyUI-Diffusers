from dataclasses import dataclass

import torch


@dataclass
class Conditioning:
    conditioning: torch.Tensor
    path: str
