import os

import folder_paths  # pyright: ignore[reportMissingImports]

from ..domain.model import Pipeline
from ..service import PipelineService


class PipelineHandler:
    def __init__(self, pipeline_service: PipelineService) -> None:
        self.create_pipeline_service = pipeline_service

    def create(self, checkpoint_name: str) -> Pipeline:
        checkpoint_path = folder_paths.get_full_path("checkpoints", checkpoint_name)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_name}")
        return self.create_pipeline_service.create_from_checkpoint(checkpoint_path)
