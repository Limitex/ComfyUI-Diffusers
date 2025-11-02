import folder_paths  # pyright: ignore[reportMissingImports]
from dependency_injector.wiring import Provide, inject

from ..di import Container
from ..ui import AutoencoderHandler
from .dto import ComfyUIAutoencoderDTO


class DiffusersVaeLoader:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[tuple[str, ...], ...]]]:
        return {
            "required": {
                "vae_name": (folder_paths.get_filename_list("vae"),),
            },
        }

    RETURN_TYPES = (ComfyUIAutoencoderDTO.COMFY_TYPE,)
    FUNCTION = "execute"
    CATEGORY = "Diffusers"

    @inject
    def execute(
        self,
        vae_name: str,
        handler: AutoencoderHandler = Provide[Container.autoencoder_handler],
    ) -> tuple[ComfyUIAutoencoderDTO]:
        vae_path = folder_paths.get_full_path("vae", vae_name)
        autoencoder_model = handler.create(vae_path)
        autoencoder_dto = ComfyUIAutoencoderDTO.from_domain(autoencoder_model)
        return (autoencoder_dto,)
