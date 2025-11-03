from dataclasses import dataclass

import PIL.Image


@dataclass
class Image:
    image: PIL.Image.Image
