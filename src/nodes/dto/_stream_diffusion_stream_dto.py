from dataclasses import dataclass
from typing import Any, ClassVar

from ...domain.model import StreamDiffusionStream


@dataclass
class ComfyUIStreamDTO:
    """Data Transfer Object for Stream Diffusion Stream in ComfyUI.

    Attributes:
        stream: StreamDiffusion instance from streamdiffusion library

    Class Attributes:
        COMFY_TYPE: Type name used in ComfyUI's RETURN_TYPES and INPUT_TYPES
    """

    COMFY_TYPE: ClassVar[str] = "DIFFUSERS_STREAM"

    stream: Any

    @classmethod
    def from_domain(cls, stream: StreamDiffusionStream) -> "ComfyUIStreamDTO":
        """Create DTO from domain model.

        Args:
            stream: StreamDiffusionStream from domain layer

        Returns:
            ComfyUIStreamDTO instance
        """
        return cls(stream=stream.stream)

    @classmethod
    def to_domain(cls, dto: "ComfyUIStreamDTO") -> StreamDiffusionStream:
        """Convert DTO back to domain model.

        Args:
            dto: ComfyUIStreamDTO instance

        Returns:
            StreamDiffusionStream domain model instance
        """
        return StreamDiffusionStream(stream=dto.stream)


@dataclass
class ComfyUIWarmupStreamDTO:
    """Data Transfer Object for Warmed-up Stream Diffusion Stream in ComfyUI.

    This is a separate type to distinguish warmed-up streams from fresh streams.

    Attributes:
        stream: StreamDiffusion instance from streamdiffusion library

    Class Attributes:
        COMFY_TYPE: Type name used in ComfyUI's RETURN_TYPES and INPUT_TYPES
    """

    COMFY_TYPE: ClassVar[str] = "DIFFUSERS_WARMUP_STREAM"

    stream: Any

    @classmethod
    def from_domain(cls, stream: StreamDiffusionStream) -> "ComfyUIWarmupStreamDTO":
        """Create DTO from domain model.

        Args:
            stream: StreamDiffusionStream from domain layer

        Returns:
            ComfyUIWarmupStreamDTO instance
        """
        return cls(stream=stream.stream)

    @classmethod
    def to_domain(cls, dto: "ComfyUIWarmupStreamDTO") -> StreamDiffusionStream:
        """Convert DTO back to domain model.

        Args:
            dto: ComfyUIWarmupStreamDTO instance

        Returns:
            StreamDiffusionStream domain model instance
        """
        return StreamDiffusionStream(stream=dto.stream)
