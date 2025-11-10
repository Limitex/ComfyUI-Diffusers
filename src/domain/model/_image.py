from dataclasses import dataclass

from PIL import Image as PilImage


@dataclass(frozen=True)
class Image:
    image: PilImage.Image

    def __post_init__(self) -> None:
        if not isinstance(self.image, PilImage.Image):
            raise ValueError(f"Image must be a PIL.Image.Image, got {type(self.image).__name__}")
