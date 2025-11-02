from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from transformers import CLIPTextModel, CLIPTokenizer


class TextEncoderRepository(ABC):
    @abstractmethod
    def encode(
        self, tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, text: str
    ) -> torch.Tensor:
        pass
