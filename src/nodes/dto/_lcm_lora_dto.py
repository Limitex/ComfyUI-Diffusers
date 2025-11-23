from dataclasses import dataclass
from typing import ClassVar

import torch

from ...domain.model import LcmLora


@dataclass
class ComfyUILcmLoraDTO:
    """Data Transfer Object for LCM LoRA in ComfyUI.

    Attributes:
        weights: Dictionary containing the LCM LoRA weights

    Class Attributes:
        COMFY_TYPE: Type name used in ComfyUI's RETURN_TYPES and INPUT_TYPES
    """

    COMFY_TYPE: ClassVar[str] = "DIFFUSERS_LCM_LORA"

    weights: dict[str, torch.Tensor]

    @classmethod
    def from_domain(cls, lcm_lora: LcmLora) -> "ComfyUILcmLoraDTO":
        """Create DTO from domain model.

        Args:
            lcm_lora: LcmLora from domain layer

        Returns:
            ComfyUILcmLoraDTO instance
        """
        return cls(weights=lcm_lora.weights)

    @classmethod
    def to_domain(cls, dto: "ComfyUILcmLoraDTO") -> LcmLora:
        """Convert DTO back to domain model.

        Args:
            dto: ComfyUILcmLoraDTO instance

        Returns:
            LcmLora domain model instance
        """
        return LcmLora(weights=dto.weights)
