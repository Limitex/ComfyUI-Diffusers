"""Dependency injection container configuration."""

from dependency_injector import containers, providers

from ..infra.diffusers import (
    DiffusersAutoencoderRepository,
    DiffusersPipelineRepository,
)
from ..infra.diffusers._text_encoder_repository import DiffusersTextEncoderRepository
from ..service import AutoencoderService, PipelineService
from ..service._clip_text_encode_service import ClipTextEncodeService
from ..ui import AutoencoderHandler, ClipTextEncodeHandler, PipelineHandler


class Container(containers.DeclarativeContainer):
    """Dependency injection container for the application."""

    # 1. Repositories
    pipeline_repository = providers.Factory(DiffusersPipelineRepository)
    autoencoder_repository = providers.Factory(DiffusersAutoencoderRepository)
    text_encoder_repository = providers.Factory(DiffusersTextEncoderRepository)

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
