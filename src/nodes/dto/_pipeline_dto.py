from dataclasses import dataclass
from typing import ClassVar

from diffusers import StableDiffusionPipeline

from ...domain.model import PipelineModel


@dataclass
class ComfyUIPipelineDTO:
    """Data Transfer Object for Diffusers Pipeline in ComfyUI.

    This DTO is designed to be used across different custom nodes.
    Other node developers can import and use this type for pipeline operations.

    Attributes:
        pipeline: StableDiffusionPipeline instance from diffusers library
        path: Path to the loaded model directory

    Class Attributes:
        COMFY_TYPE: Type name used in ComfyUI's RETURN_TYPES and INPUT_TYPES

    Example:
        ```python
        from your_node_package.nodes.dto import ComfyUIPipelineDTO

        class YourCustomNode:
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"pipeline": (ComfyUIPipelineDTO.COMFY_TYPE,)}}

            RETURN_TYPES = (ComfyUIPipelineDTO.COMFY_TYPE,)

            def execute(self, pipeline: ComfyUIPipelineDTO):
                result = pipeline.pipeline(prompt="...")
                return (result,)
        ```
    """

    # ClassVar to avoid being treated as a dataclass field
    COMFY_TYPE: ClassVar[str] = "DIFFUSERS_PIPELINE"

    pipeline: StableDiffusionPipeline
    path: str

    @classmethod
    def from_domain(cls, model: PipelineModel) -> "ComfyUIPipelineDTO":
        """Create DTO from domain model.

        Args:
            model: PipelineModel from domain layer

        Returns:
            ComfyUIPipelineDTO instance
        """
        return cls(pipeline=model.pipeline, path=model.path)
