from dataclasses import dataclass
from typing import ClassVar

import torch

from ...domain.model import Conditioning


@dataclass
class ComfyUIConditioningDTO:
    """Data Transfer Object for Diffusers Conditioning in ComfyUI.

    This DTO is designed to be used across different custom nodes.
    Other node developers can import and use this type for conditioning operations.

    Attributes:
        conditioning: Tensor representing text embeddings/conditioning
        path: Path to the source model directory

    Class Attributes:
        COMFY_TYPE: Type name used in ComfyUI's RETURN_TYPES and INPUT_TYPES

    Example:
        ```python
        from your_node_package.nodes.dto import ComfyUIConditioningDTO

        class YourCustomNode:
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"conditioning": (ComfyUIConditioningDTO.COMFY_TYPE,)}}

            RETURN_TYPES = (ComfyUIConditioningDTO.COMFY_TYPE,)

            def execute(self, conditioning: ComfyUIConditioningDTO):
                # Use the conditioning tensor for generation
                result = process_with_conditioning(conditioning.conditioning)
                return (result,)
        ```
    """

    # ClassVar to avoid being treated as a dataclass field
    COMFY_TYPE: ClassVar[str] = "DIFFUSERS_CONDITIONING"

    conditioning: torch.Tensor
    path: str

    @classmethod
    def from_domain(cls, conditioning: Conditioning) -> "ComfyUIConditioningDTO":
        """Create DTO from domain model.

        Args:
            conditioning: Conditioning from domain layer

        Returns:
            ComfyUIConditioningDTO instance
        """
        return cls(conditioning=conditioning.conditioning, path=conditioning.path)
