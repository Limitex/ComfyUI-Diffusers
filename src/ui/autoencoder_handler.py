import os

import folder_paths  # pyright: ignore[reportMissingImports]

from ..domain.model import AutoencoderModel
from ..service.create_autoencoder_service import CreateAutoencoderService


class AutoencoderHandler:
    def __init__(self, create_autoencoder_service: CreateAutoencoderService) -> None:
        self.create_autoencoder_service = create_autoencoder_service

    def create(self, checkpoint_name: str) -> AutoencoderModel:
        checkpoint_path = folder_paths.get_full_path("vae", checkpoint_name)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"VAE file not found: {checkpoint_name}")
        return self.create_autoencoder_service.create_from_checkpoint(checkpoint_path)
