"""Dependency injection container configuration."""

from dependency_injector import containers, providers

from ..infra.diffusers import (
    DiffusersAutoencoderRepository,
    DiffusersPipelineRepository,
    DiffusersSamplerRepository,
    DiffusersSchedulerRepository,
    DiffusersTextEncoderRepository,
)
from ..usecase import (
    AutoencoderUsecase,
    ClipTextEncodeUsecase,
    PipelineUsecase,
    SamplerUsecase,
    SchedulerUsecase,
)


class Container(containers.DeclarativeContainer):
    """Dependency injection container for the application."""

    # 1. Repositories
    pipeline_repository = providers.Factory(DiffusersPipelineRepository)
    autoencoder_repository = providers.Factory(DiffusersAutoencoderRepository)
    text_encoder_repository = providers.Factory(DiffusersTextEncoderRepository)
    sampler_repository = providers.Factory(DiffusersSamplerRepository)
    scheduler_repository = providers.Factory(DiffusersSchedulerRepository)

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
