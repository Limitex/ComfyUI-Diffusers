from abc import ABC, abstractmethod

import torch


class LcmLoraRepository(ABC):
    @abstractmethod
    def load_lcm_lora(self, lora_path: str) -> dict[str, torch.Tensor]:
        """Load LCM LoRA weights from file.

        Args:
            lora_path: Path to the LCM LoRA file

        Returns:
            Dictionary containing the loaded weights
        """
