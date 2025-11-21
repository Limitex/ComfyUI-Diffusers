import folder_paths  # pyright: ignore[reportMissingImports]
from dependency_injector.wiring import Provide, inject

from ..di import Container
from ..usecase import LcmLoraUsecase
from .dto import ComfyUILcmLoraDTO


class LcmLoraLoader:
    """Node to load LCM LoRA weights for Stream Diffusion."""

    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[tuple[str, ...], ...]]]:
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"),),
            }
        }

    RETURN_TYPES = (ComfyUILcmLoraDTO.COMFY_TYPE,)
    FUNCTION = "execute"
    CATEGORY = "Diffusers/StreamDiffusion"

    @inject
    def execute(
        self,
        lora_name: str,
        usecase: LcmLoraUsecase = Provide[Container.lcm_lora_usecase],
    ) -> tuple[ComfyUILcmLoraDTO]:
        """Load LCM LoRA weights.

        Args:
            lora_name: Name of the LoRA file
            usecase: Injected LcmLoraUsecase

        Returns:
            Tuple containing ComfyUILcmLoraDTO
        """
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lcm_lora = usecase.execute(lora_path)
        dto = ComfyUILcmLoraDTO.from_domain(lcm_lora)
        return (dto,)
