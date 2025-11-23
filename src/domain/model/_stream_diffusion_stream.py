from dataclasses import dataclass

from streamdiffusion import StreamDiffusion  # type: ignore[import-untyped]


@dataclass(frozen=True)
class StreamDiffusionStream:
    """Stream Diffusion stream instance."""

    stream: StreamDiffusion

    def __post_init__(self) -> None:
        if self.stream is None:
            raise ValueError("Stream cannot be None")
        # We can't check for exact type here as it would create circular dependency
        # The actual type checking will be done at runtime in the repository layer
