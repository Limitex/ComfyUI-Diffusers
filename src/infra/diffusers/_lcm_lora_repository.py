from typing import Any

from safetensors.torch import load_file

from ...domain.repositories import LcmLoraRepository


class DiffusersLcmLoraRepository(LcmLoraRepository):
    def load_lcm_lora(self, lora_path: str) -> dict[str, Any]:
        """Load LCM LoRA weights from safetensors file.

        Args:
            lora_path: Path to the LCM LoRA safetensors file

        Returns:
            Dictionary containing the loaded weights
        """
        return load_file(lora_path)
