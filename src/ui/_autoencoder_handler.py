import os

import folder_paths  # pyright: ignore[reportMissingImports]

from ..domain.model import Autoencoder
from ..service import AutoencoderService


class AutoencoderHandler:
    def __init__(self, autoencoder_service: AutoencoderService) -> None:
        self.create_autoencoder_service = autoencoder_service

    def create(self, checkpoint_name: str) -> Autoencoder:
        checkpoint_path = folder_paths.get_full_path("vae", checkpoint_name)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"VAE file not found: {checkpoint_name}")
        return self.create_autoencoder_service.create_from_checkpoint(checkpoint_path)
