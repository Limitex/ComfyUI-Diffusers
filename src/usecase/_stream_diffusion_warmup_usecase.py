from ..domain.model import CFGScale, Delta, Seed, Steps, StreamDiffusionStream, WarmupCount
from ..domain.repositories import StreamDiffusionRepository


class StreamDiffusionWarmupUsecase:
    def __init__(self, stream_diffusion_repo: StreamDiffusionRepository) -> None:
        self.stream_diffusion_repo = stream_diffusion_repo

    def execute(
        self,
        stream: StreamDiffusionStream,
        negative_prompt: str,
        steps: Steps,
        cfg: CFGScale,
        delta: Delta,
        seed: Seed,
        warmup_count: WarmupCount,
    ) -> StreamDiffusionStream:
        """Warm up a Stream Diffusion stream.

        Args:
            stream: Stream to warm up
            negative_prompt: Negative prompt text
            steps: Number of inference steps
            cfg: CFG scale value
            delta: Delta value
            seed: Random seed
            warmup_count: Number of warmup iterations

        Returns:
            Warmed up StreamDiffusionStream (same instance)
        """
        # Prepare stream with parameters
        self.stream_diffusion_repo.prepare_stream(
            stream=stream.stream,
            prompt="",
            negative_prompt=negative_prompt,
            steps=steps,
            cfg=cfg,
            delta=delta,
            seed=seed,
        )

        # Warmup (txt2img mode - no input image)
        self.stream_diffusion_repo.warmup_stream(
            stream=stream.stream,
            warmup_count=warmup_count,
            input_image=None,
        )

        return stream
