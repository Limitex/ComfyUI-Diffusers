import os

import torch
from comfy.model_management import get_torch_device  # pyright: ignore[reportMissingImports]
from diffusers import AutoencoderKL

from ...domain.repositories import AutoencoderRepository
from ...utils import get_cache_path
from ._cache import is_vae_cached


class DiffusersAutoencoderRepository(AutoencoderRepository):
    def __init__(self) -> None:
        self.cache_dir = get_cache_path()
        self.device = get_torch_device()

    def convert_and_save_from_single_file(self, checkpoint_path: str, dtype: torch.dtype) -> str:
        checkpoint_name = os.path.basename(checkpoint_path)
        ckpt_cache_path = os.path.join(self.cache_dir, checkpoint_name)

        # Check if already cached
        if is_vae_cached(ckpt_cache_path):
            return ckpt_cache_path

        AutoencoderKL.from_single_file(
            pretrained_model_link_or_path=checkpoint_path,
            torch_dtype=dtype,
            cache_dir=self.cache_dir,
        ).save_pretrained(ckpt_cache_path, safe_serialization=True)
        return ckpt_cache_path

    def load_autoencoder_from_path(self, model_path: str, dtype: torch.dtype) -> AutoencoderKL:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")
        vae: AutoencoderKL = AutoencoderKL.from_pretrained(  # type: ignore[no-untyped-call]
            pretrained_model_name_or_path=model_path,
            torch_dtype=dtype,
            cache_dir=self.cache_dir,
        ).to(self.device)
        if vae is None:
            raise RuntimeError(f"Failed to load AutoencoderKL from path: {model_path}")
        return vae
