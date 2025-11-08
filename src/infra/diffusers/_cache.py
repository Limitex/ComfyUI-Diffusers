import os
from typing import Final

# Centralized cache helpers for diffusers-related repositories

_PIPELINE_INDEX_FILE: Final[str] = "model_index.json"
_VAE_CONFIG_FILE: Final[str] = "config.json"
_VAE_WEIGHT_CANDIDATES: Final[tuple[str, ...]] = (
    "diffusion_pytorch_model.safetensors",
    "diffusion_pytorch_model.bin",
    "pytorch_model.bin",
    "model.safetensors",
)


def is_pipeline_cached(cache_path: str) -> bool:
    """Returns True if the Stable Diffusion pipeline has already been converted.

    Criteria: directory exists and contains a `model_index.json`.
    """
    return os.path.isdir(cache_path) and os.path.isfile(
        os.path.join(cache_path, _PIPELINE_INDEX_FILE)
    )


def is_vae_cached(cache_path: str) -> bool:
    """Returns True if the VAE has already been converted.

    Criteria: directory exists, contains `config.json`, and at least one known weight file.
    """
    if not (
        os.path.isdir(cache_path) and os.path.isfile(os.path.join(cache_path, _VAE_CONFIG_FILE))
    ):
        return False
    return any(os.path.isfile(os.path.join(cache_path, fname)) for fname in _VAE_WEIGHT_CANDIDATES)
