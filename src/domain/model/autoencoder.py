from dataclasses import dataclass

from diffusers import AutoencoderKL


@dataclass
class AutoencoderModel:
    autoencoder: AutoencoderKL
    path: str
