"""Dependency injection container configuration."""

from dependency_injector import containers, providers

from ..infra.diffusers import (
    DiffusersAutoencoderRepository,
    DiffusersLcmLoraRepository,
    DiffusersPipelineRepository,
    DiffusersSamplerRepository,
    DiffusersSchedulerRepository,
    DiffusersStreamDiffusionRepository,
    DiffusersTextEncoderRepository,
)
from ..usecase import (
    AutoencoderUsecase,
    ClipTextEncodeUsecase,
    LcmLoraUsecase,
    PipelineUsecase,
    SamplerUsecase,
    SchedulerUsecase,
    StreamDiffusionCreateStreamUsecase,
    StreamDiffusionFastSampleUsecase,
    StreamDiffusionSampleUsecase,
    StreamDiffusionWarmupUsecase,
)


class Container(containers.DeclarativeContainer):
    """Dependency injection container for the application."""

    # 1. Repositories
    pipeline_repository = providers.Factory(DiffusersPipelineRepository)
    autoencoder_repository = providers.Factory(DiffusersAutoencoderRepository)
    text_encoder_repository = providers.Factory(DiffusersTextEncoderRepository)
    sampler_repository = providers.Factory(DiffusersSamplerRepository)
    scheduler_repository = providers.Factory(DiffusersSchedulerRepository)
    lcm_lora_repository = providers.Factory(DiffusersLcmLoraRepository)
    stream_diffusion_repository = providers.Factory(DiffusersStreamDiffusionRepository)

    # 2. Usecases
    pipeline_usecase = providers.Factory(
        PipelineUsecase,
        pipeline_repo=pipeline_repository,
    )
    autoencoder_usecase = providers.Factory(
        AutoencoderUsecase,
        autoencoder_repo=autoencoder_repository,
    )
    clip_text_encode_usecase = providers.Factory(
        ClipTextEncodeUsecase,
        text_encoder_repo=text_encoder_repository,
    )
    sampler_usecase = providers.Factory(
        SamplerUsecase,
        sampler_repo=sampler_repository,
    )
    scheduler_usecase = providers.Factory(
        SchedulerUsecase,
        scheduler_repo=scheduler_repository,
    )
    lcm_lora_usecase = providers.Factory(
        LcmLoraUsecase,
        lcm_lora_repo=lcm_lora_repository,
    )
    stream_diffusion_create_stream_usecase = providers.Factory(
        StreamDiffusionCreateStreamUsecase,
        stream_diffusion_repo=stream_diffusion_repository,
    )
    stream_diffusion_warmup_usecase = providers.Factory(
        StreamDiffusionWarmupUsecase,
        stream_diffusion_repo=stream_diffusion_repository,
    )
    stream_diffusion_sample_usecase = providers.Factory(
        StreamDiffusionSampleUsecase,
        stream_diffusion_repo=stream_diffusion_repository,
    )
    stream_diffusion_fast_sample_usecase = providers.Factory(
        StreamDiffusionFastSampleUsecase,
        stream_diffusion_repo=stream_diffusion_repository,
    )
