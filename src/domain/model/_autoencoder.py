from dataclasses import dataclass

from diffusers import AutoencoderKL


@dataclass
class Autoencoder:
    autoencoder: AutoencoderKL
    path: str
