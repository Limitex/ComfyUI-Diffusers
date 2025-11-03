"""Dependency injection container configuration."""

from dependency_injector import containers, providers

from ..infra.diffusers import (
    DiffusersAutoencoderRepository,
    DiffusersPipelineRepository,
    DiffusersSamplingRepository,
    DiffusersTextEncoderRepository,
)
from ..service import AutoencoderService, ClipTextEncodeService, PipelineService, SamplingService
from ..ui import AutoencoderHandler, ClipTextEncodeHandler, PipelineHandler, SamplerHandler


class Container(containers.DeclarativeContainer):
    """Dependency injection container for the application."""

    # 1. Repositories
    pipeline_repository = providers.Factory(DiffusersPipelineRepository)
    autoencoder_repository = providers.Factory(DiffusersAutoencoderRepository)
    text_encoder_repository = providers.Factory(DiffusersTextEncoderRepository)
    sampling_repository = providers.Factory(DiffusersSamplingRepository)

    # 2. Services
    pipeline_service = providers.Factory(
        PipelineService,
        pipeline_repo=pipeline_repository,
    )
    autoencoder_service = providers.Factory(
        AutoencoderService,
        autoencoder_repo=autoencoder_repository,
    )
    clip_text_encode_service = providers.Factory(
        ClipTextEncodeService,
        text_encoder_repo=text_encoder_repository,
    )
    sampling_service = providers.Factory(
        SamplingService,
        sampling_repo=sampling_repository,
    )

    # 3. Handler
    pipeline_handler = providers.Factory(
        PipelineHandler,
        pipeline_service=pipeline_service,
    )
    autoencoder_handler = providers.Factory(
        AutoencoderHandler,
        autoencoder_service=autoencoder_service,
    )
    clip_text_encode_handler = providers.Factory(
        ClipTextEncodeHandler,
        clip_text_encode_service=clip_text_encode_service,
    )
    sampler_handler = providers.Factory(
        SamplerHandler,
        sampling_service=sampling_service,
    )
