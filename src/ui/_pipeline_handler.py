import os

from ..domain.model import Clip, Pipeline
from ..service import PipelineService


class PipelineHandler:
    def __init__(self, pipeline_service: PipelineService) -> None:
        self.create_pipeline_service = pipeline_service

    def create(self, checkpoint_path: str) -> tuple[Pipeline, Clip]:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        pipeline, clip = self.create_pipeline_service.create_from_checkpoint(checkpoint_path)
        if pipeline is None:
            raise RuntimeError(f"Failed to create pipeline from checkpoint: {checkpoint_path}")
        if clip is None:
            raise RuntimeError(f"Failed to extract clip from checkpoint: {checkpoint_path}")
        return pipeline, clip
