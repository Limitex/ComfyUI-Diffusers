import os

from ..domain.model import Autoencoder
from ..service import AutoencoderService


class AutoencoderHandler:
    def __init__(self, autoencoder_service: AutoencoderService) -> None:
        self.create_autoencoder_service = autoencoder_service

    def create(self, vae_path: str) -> Autoencoder:
        if not os.path.exists(vae_path):
            raise FileNotFoundError(f"VAE file not found: {vae_path}")
        autoencoder = self.create_autoencoder_service.create_from_checkpoint(vae_path)
        if autoencoder is None:
            raise RuntimeError(f"Failed to create autoencoder from checkpoint: {vae_path}")
        return autoencoder
