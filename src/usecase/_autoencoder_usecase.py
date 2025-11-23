import os

import torch

from ..domain.model import Autoencoder
from ..domain.repositories import AutoencoderRepository


class AutoencoderUsecase:
    def __init__(self, autoencoder_repo: AutoencoderRepository) -> None:
        self.autoencoder_repo = autoencoder_repo
        self.dtype = torch.float16

    def execute(self, vae_path: str) -> Autoencoder:
        if not os.path.exists(vae_path):
            raise FileNotFoundError(f"VAE file not found: {vae_path}")

        path = self.autoencoder_repo.convert_and_save_from_single_file(vae_path, self.dtype)
        if not os.path.exists(path):
            raise RuntimeError(f"Failed to convert VAE checkpoint: {vae_path}")

        vae = self.autoencoder_repo.load_autoencoder_from_path(path, self.dtype)
        if vae is None:
            raise RuntimeError(f"Failed to create autoencoder from checkpoint: {vae_path}")

        return Autoencoder(autoencoder=vae, path=path)
