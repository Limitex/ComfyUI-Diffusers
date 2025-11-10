from dependency_injector.wiring import Provide, inject

from ..di import Container
from ..usecase import ClipTextEncodeUsecase
from .dto import ComfyUIClipDTO, ComfyUIConditioningDTO


class DiffusersClipTextEncode:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[str, ...] | tuple[str, dict[str, bool]]]]:
        return {
            "required": {
                "clip": (ComfyUIClipDTO.COMFY_TYPE,),
                "text": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = (ComfyUIConditioningDTO.COMFY_TYPE,)
    FUNCTION = "execute"
    CATEGORY = "Diffusers"

    @inject
    def execute(
        self,
        clip: ComfyUIClipDTO,
        text: str,
        usecase: ClipTextEncodeUsecase = Provide[Container.clip_text_encode_usecase],
    ) -> tuple[ComfyUIConditioningDTO]:
        clip_domain = ComfyUIClipDTO.to_domain(clip)
        conditioning_model = usecase.execute(
            clip_domain.tokenizer,
            clip_domain.text_encoder,
            clip_domain.path,
            text,
        )
        conditioning_dto = ComfyUIConditioningDTO.from_domain(conditioning_model)
        return (conditioning_dto,)
