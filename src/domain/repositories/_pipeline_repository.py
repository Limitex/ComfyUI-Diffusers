from abc import ABC, abstractmethod

import torch
from diffusers import StableDiffusionPipeline


class PipelineRepository(ABC):
    @abstractmethod
    def convert_and_save_from_single_file(self, checkpoint_path: str, dtype: torch.dtype) -> str:
        pass

    @abstractmethod
    def load_pipeline_from_path(
        self, model_path: str, dtype: torch.dtype
    ) -> StableDiffusionPipeline:
        pass
