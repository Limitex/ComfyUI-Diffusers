from dataclasses import dataclass
from enum import Enum


class CFGTypeEnum(str, Enum):
    """CFG type options for Stream Diffusion."""

    NONE = "none"
    FULL = "full"
    SELF = "self"
    INITIALIZE = "initialize"


@dataclass(frozen=True)
class CFGType:
    """CFG type configuration for Stream Diffusion."""

    value: CFGTypeEnum

    def __post_init__(self) -> None:
        if not isinstance(self.value, CFGTypeEnum):
            raise ValueError(f"CFG type must be a CFGTypeEnum, got {type(self.value).__name__}")
