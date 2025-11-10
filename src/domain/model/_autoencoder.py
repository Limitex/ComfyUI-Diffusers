from dataclasses import dataclass

from diffusers import AutoencoderKL


@dataclass(frozen=True)
class Autoencoder:
    autoencoder: AutoencoderKL
    path: str

    def __post_init__(self) -> None:
        if not isinstance(self.autoencoder, AutoencoderKL):
            raise ValueError(
                f"Autoencoder must be an instance of AutoencoderKL, got {type(self.autoencoder).__name__}"
            )
        if not isinstance(self.path, str):
            raise ValueError(f"Path must be a string, got {type(self.path).__name__}")
        if not self.path.strip():
            raise ValueError("Path cannot be empty or whitespace only")
