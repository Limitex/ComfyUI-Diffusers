from dataclasses import dataclass
from typing import TypeAlias

import torch
from torchvision.transforms import ToTensor

from ....domain.model import Image

ComfyUIImageDTO: TypeAlias = torch.Tensor


@dataclass
class ComfyUIImage:
    COMFY_TYPE: str = "IMAGE"

    @classmethod
    def from_domain(cls, image: Image) -> "ComfyUIImageDTO":
        tensor: torch.Tensor = ToTensor()(image.image)
        return tensor.permute(1, 2, 0)

    @classmethod
    def from_domains(cls, images: list[Image]) -> "ComfyUIImageDTO":
        return torch.stack([cls.from_domain(image) for image in images])
