from __future__ import annotations

from transformers import CLIPTextModel, CLIPTokenizer

from ..domain.model import Conditioning
from ..domain.repositories._text_encoder_repository import TextEncoderRepository


class ClipTextEncodeUsecase:
    def __init__(self, text_encoder_repo: TextEncoderRepository) -> None:
        self.text_encoder_repo = text_encoder_repo

    def execute(
        self,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModel,
        path: str,
        text: str,
    ) -> Conditioning:
        if not text.strip():
            raise ValueError("Text input is empty or whitespace only.")

        embeddings = self.text_encoder_repo.encode(tokenizer, text_encoder, text)
        if embeddings is None:
            raise RuntimeError("Failed to generate embeddings.")

        return Conditioning(conditioning=embeddings, path=path)
