import os

import folder_paths  # pyright: ignore[reportMissingImports]

from ..domain.model import PipelineModel
from ..service.create_pipeline_service import CreatePipelineService


class PipelineHandler:
    def __init__(self, create_pipeline_service: CreatePipelineService) -> None:
        self.create_pipeline_service = create_pipeline_service

    def create(self, checkpoint_name: str) -> PipelineModel:
        checkpoint_path = folder_paths.get_full_path("checkpoints", checkpoint_name)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_name}")
        return self.create_pipeline_service.create_from_checkpoint(checkpoint_path)
