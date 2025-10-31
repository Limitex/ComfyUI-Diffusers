"""Dependency injection container configuration."""

from dependency_injector import containers, providers

from ..infra.diffusers import DiffusersAutoencoderRepository, DiffusersPipelineRepository
from ..service import AutoencoderService, PipelineService
from ..ui import AutoencoderHandler, PipelineHandler


class Container(containers.DeclarativeContainer):
    """Dependency injection container for the application."""

    # 1. Repositories
    pipeline_repository = providers.Factory(DiffusersPipelineRepository)
    autoencoder_repository = providers.Factory(DiffusersAutoencoderRepository)

    # 2. Services
    pipeline_service = providers.Factory(
        PipelineService,
        pipeline_repo=pipeline_repository,
    )
    autoencoder_service = providers.Factory(
        AutoencoderService,
        autoencoder_repo=autoencoder_repository,
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
