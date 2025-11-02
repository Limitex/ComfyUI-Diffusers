from dataclasses import dataclass
from typing import ClassVar

from transformers import CLIPTextModel, CLIPTokenizer

from ...domain.model import Clip


@dataclass
class ComfyUIClipDTO:
    """Data Transfer Object for CLIP (Tokenizer and Text Encoder) in ComfyUI.

    This DTO is designed to be used across different custom nodes.
    Other node developers can import and use this type for CLIP operations.

    Attributes:
        tokenizer: CLIPTokenizer instance from transformers library
        text_encoder: CLIPTextModel instance from transformers library
        path: Path to the loaded model directory

    Class Attributes:
        COMFY_TYPE: Type name used in ComfyUI's RETURN_TYPES and INPUT_TYPES

    Example:
        ```python
        from your_node_package.nodes.dto import ComfyUIClipDTO

        class YourCustomNode:
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"clip": (ComfyUIClipDTO.COMFY_TYPE,)}}

            RETURN_TYPES = (ComfyUIClipDTO.COMFY_TYPE,)

            def execute(self, clip: ComfyUIClipDTO):
                # Use the tokenizer and text encoder
                tokens = clip.tokenizer(prompt, return_tensors="pt")
                embeddings = clip.text_encoder(**tokens)
                return (embeddings,)
        ```
    """

    # ClassVar to avoid being treated as a dataclass field
    COMFY_TYPE: ClassVar[str] = "DIFFUSERS_CLIP"

    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModel
    path: str

    @classmethod
    def from_domain(cls, clip: Clip) -> "ComfyUIClipDTO":
        """Create DTO from domain model.

        Args:
            clip: Clip from domain layer

        Returns:
            ComfyUIClipDTO instance
        """
        return cls(tokenizer=clip.tokenizer, text_encoder=clip.text_encoder, path=clip.path)
