import os

import torch

from ..domain.model import Pipeline
from ..domain.repositories import PipelineRepository


class PipelineService:
    def __init__(self, pipeline_repo: PipelineRepository) -> None:
        self.pipeline_repo = pipeline_repo
        self.dtype = torch.float32

    def create_from_checkpoint(self, checkpoint_full_path: str) -> Pipeline:
        path = self.pipeline_repo.convert_and_save_from_single_file(
            checkpoint_full_path, self.dtype
        )
        if not os.path.exists(path):
            raise RuntimeError(f"Failed to convert pipeline checkpoint: {checkpoint_full_path}")

        pipe = self.pipeline_repo.load_pipeline_from_path(path, self.dtype)
        if pipe is None:
            raise RuntimeError(f"Failed to create pipeline from checkpoint: {checkpoint_full_path}")

        return Pipeline(pipeline=pipe, path=path)
