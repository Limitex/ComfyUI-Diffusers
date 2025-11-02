import os

import folder_paths  # pyright: ignore[reportMissingImports]
import torch
from comfy.model_management import get_torch_device  # pyright: ignore[reportMissingImports]
from diffusers import StableDiffusionPipeline

from ...domain.repositories import PipelineRepository


class DiffusersPipelineRepository(PipelineRepository):
    def __init__(self) -> None:
        self.cache_dir = folder_paths.get_temp_directory()
        self.device = get_torch_device()

    def convert_and_save_from_single_file(self, checkpoint_path: str, dtype: torch.dtype) -> str:
        checkpoint_name = os.path.basename(checkpoint_path)
        ckpt_cache_path = os.path.join(self.cache_dir, checkpoint_name)

        StableDiffusionPipeline.from_single_file(
            pretrained_model_link_or_path=checkpoint_path,
            torch_dtype=dtype,
            cache_dir=self.cache_dir,
        ).save_pretrained(ckpt_cache_path, safe_serialization=True)
        return ckpt_cache_path

    def load_pipeline_from_path(
        self, model_path: str, dtype: torch.dtype
    ) -> StableDiffusionPipeline:
        pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(  # type: ignore[no-untyped-call]
            pretrained_model_name_or_path=model_path,
            torch_dtype=dtype,
            cache_dir=self.cache_dir,
        ).to(self.device)
        return pipe
