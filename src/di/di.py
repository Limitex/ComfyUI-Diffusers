from dependency_injector import containers, providers

from ..infra.diffusers import DiffusersAutoencoderRepository
from ..infra.diffusers import DiffusersPipelineRepository
from ..service import CreateAutoencoderService
from ..service import CreatePipelineService
from ..ui import AutoencoderHandler
from ..ui import PipelineHandler


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
