from __future__ import annotations

from typing import Any

import torch
from comfy.model_management import get_torch_device  # pyright: ignore[reportMissingImports]
from transformers import CLIPTextModel, CLIPTokenizer

from ...domain.repositories._text_encoder_repository import TextEncoderRepository


class DiffusersTextEncoderRepository(TextEncoderRepository):
    def __init__(self) -> None:
        self.device = get_torch_device()

    def encode(
        self, tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, text: str
    ) -> torch.Tensor:
        max_length = tokenizer.model_max_length
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        text_length = input_ids.shape[-1]

        if max_length < text_length:
            text_ids = input_ids.to(self.device)
        else:
            text_ids = tokenizer(
                text,
                truncation=False,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            ).input_ids.to(self.device)

        concat_embeds: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, text_ids.shape[-1], max_length):
                segment_ids = text_ids[:, i : i + max_length]
                outputs: Any = text_encoder(segment_ids)
                if hasattr(outputs, "last_hidden_state"):
                    embeds = outputs.last_hidden_state
                else:
                    embeds = outputs[0]
                concat_embeds.append(embeds)

        text_embeds: torch.Tensor = torch.cat(concat_embeds, dim=1)
        return text_embeds
