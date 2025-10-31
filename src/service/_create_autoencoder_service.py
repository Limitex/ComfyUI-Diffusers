import os

import torch

from ..domain.model import AutoencoderModel
from ..domain.repositories import IAutoencoderRepository


class CreateAutoencoderService:
    def __init__(self, autoencoder_repo: IAutoencoderRepository) -> None:
        self.autoencoder_repo = autoencoder_repo
        self.dtype = torch.float32

    def create_from_checkpoint(self, checkpoint_full_path: str) -> AutoencoderModel:
        path = self.autoencoder_repo.convert_and_save_from_single_file(checkpoint_full_path)
        if not os.path.exists(path):
            raise RuntimeError(f"Failed to convert VAE checkpoint: {checkpoint_full_path}")

        vae = self.autoencoder_repo.load_autoencoder_from_path(path, self.dtype)
        if vae is None:
            raise RuntimeError(
                f"Failed to create autoencoder from checkpoint: {checkpoint_full_path}"
            )

        return AutoencoderModel(autoencoder=vae, path=path)
