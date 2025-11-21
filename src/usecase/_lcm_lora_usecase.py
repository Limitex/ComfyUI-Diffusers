from ..domain.model import LcmLora
from ..domain.repositories import LcmLoraRepository


class LcmLoraUsecase:
    def __init__(self, lcm_lora_repo: LcmLoraRepository) -> None:
        self.lcm_lora_repo = lcm_lora_repo

    def execute(self, lora_path: str) -> LcmLora:
        """Load LCM LoRA weights from file.

        Args:
            lora_path: Path to the LCM LoRA file

        Returns:
            LcmLora domain model

        Raises:
            FileNotFoundError: If the lora file doesn't exist
            RuntimeError: If loading fails
        """
        weights = self.lcm_lora_repo.load_lcm_lora(lora_path)
        if not weights:
            raise RuntimeError(f"Failed to load LCM LoRA weights from: {lora_path}")

        return LcmLora(weights=weights)
