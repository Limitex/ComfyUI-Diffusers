from dataclasses import dataclass
from typing import ClassVar

from diffusers import AutoencoderKL

from ...domain.model import Autoencoder


@dataclass
class ComfyUIAutoencoderDTO:
    """Data Transfer Object for Diffusers Autoencoder in ComfyUI.

    This DTO is designed to be used across different custom nodes.
    Other node developers can import and use this type for autoencoder operations.

    Attributes:
        autoencoder: AutoencoderKL instance from diffusers library
        path: Path to the loaded model directory

    Class Attributes:
        COMFY_TYPE: Type name used in ComfyUI's RETURN_TYPES and INPUT_TYPES

    Example:
        ```python
        from your_node_package.nodes.dto import ComfyUIAutoencoderDTO

        class YourCustomNode:
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"autoencoder": (ComfyUIAutoencoderDTO.COMFY_TYPE,)}}

            RETURN_TYPES = (ComfyUIAutoencoderDTO.COMFY_TYPE,)

            def execute(self, autoencoder: ComfyUIAutoencoderDTO):
                # Use the autoencoder for encoding/decoding
                encoded = autoencoder.autoencoder.encode(image)
                return (encoded,)
        ```
    """

    # ClassVar to avoid being treated as a dataclass field
    COMFY_TYPE: ClassVar[str] = "DIFFUSERS_VAE"

    autoencoder: AutoencoderKL
    path: str

    @classmethod
    def from_domain(cls, autoencoder: Autoencoder) -> "ComfyUIAutoencoderDTO":
        """Create DTO from domain model.

        Args:
            autoencoder: Autoencoder from domain layer

        Returns:
            ComfyUIAutoencoderDTO instance
        """
        return cls(autoencoder=autoencoder.autoencoder, path=autoencoder.path)
