from dependency_injector import containers, providers

from ..infra.diffusers.autoencoder_repository import DiffusersAutoencoderRepository
from ..infra.diffusers.pipeline_repository import DiffusersPipelineRepository
from ..service.create_autoencoder_service import CreateAutoencoderService
from ..service.create_pipeline_service import CreatePipelineService
from ..ui.autoencoder_handler import AutoencoderHandler
from ..ui.pipeline_handler import PipelineHandler


class Container(containers.DeclarativeContainer):
    # 1. Repositories
    pipeline_repository = providers.Factory(DiffusersPipelineRepository)
    autoencoder_repository = providers.Factory(DiffusersAutoencoderRepository)

    # 2. Services
    create_pipeline_service = providers.Factory(
        CreatePipelineService,
        pipeline_repo=pipeline_repository,
    )
    create_autoencoder_service = providers.Factory(
        CreateAutoencoderService,
        autoencoder_repo=autoencoder_repository,
    )

    # 3. Handler
    pipeline_handler = providers.Factory(
        PipelineHandler,
        create_pipeline_service=create_pipeline_service,
    )
    autoencoder_handler = providers.Factory(
        AutoencoderHandler,
        create_autoencoder_service=create_autoencoder_service,
    )
