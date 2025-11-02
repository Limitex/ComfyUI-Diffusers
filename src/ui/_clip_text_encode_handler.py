from __future__ import annotations

from transformers import CLIPTextModel, CLIPTokenizer

from ..domain.model import Conditioning
from ..service._clip_text_encode_service import ClipTextEncodeService


class ClipTextEncodeHandler:
    def __init__(self, clip_text_encode_service: ClipTextEncodeService) -> None:
        self.clip_text_encode_service = clip_text_encode_service

    def encode(
        self, tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, path: str, text: str
    ) -> Conditioning:
        if not text:
            raise ValueError("Text input is empty.")
        if path is None:
            raise ValueError("Clip path is None.")
        if tokenizer is None or text_encoder is None:
            raise ValueError("Tokenizer or Text Encoder is None.")
        if not isinstance(text, str):
            raise TypeError("Text input must be a string.")
        conditioning = self.clip_text_encode_service.encode(tokenizer, text_encoder, path, text)
        if conditioning is None:
            raise RuntimeError("Failed to encode text into conditioning.")
        return conditioning
