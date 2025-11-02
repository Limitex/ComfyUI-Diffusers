from dataclasses import dataclass

from transformers import CLIPTextModel, CLIPTokenizer


@dataclass
class Clip:
    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModel
    path: str
