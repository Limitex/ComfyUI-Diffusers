from abc import ABC, abstractmethod
from typing import Any


class LcmLoraRepository(ABC):
    @abstractmethod
    def load_lcm_lora(self, lora_path: str) -> dict[str, Any]:
        """Load LCM LoRA weights from file.

        Args:
            lora_path: Path to the LCM LoRA file

        Returns:
            Dictionary containing the loaded weights
        """
