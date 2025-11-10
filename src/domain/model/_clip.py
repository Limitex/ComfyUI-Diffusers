from dataclasses import dataclass

from transformers import CLIPTextModel, CLIPTokenizer


@dataclass(frozen=True)
class Clip:
    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModel
    path: str

    def __post_init__(self) -> None:
        if not isinstance(self.tokenizer, CLIPTokenizer):
            raise ValueError(
                f"Tokenizer must be an instance of CLIPTokenizer, got {type(self.tokenizer).__name__}"
            )
        if not isinstance(self.text_encoder, CLIPTextModel):
            raise ValueError(
                f"Text encoder must be an instance of CLIPTextModel, got {type(self.text_encoder).__name__}"
            )
        if not isinstance(self.path, str):
            raise ValueError(f"Path must be a string, got {type(self.path).__name__}")
        if not self.path.strip():
            raise ValueError("Path cannot be empty or whitespace only")
