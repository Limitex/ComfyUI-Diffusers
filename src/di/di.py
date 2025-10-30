from dependency_injector import containers, providers

from ..infra.diffusers.pipeline_repository import DiffusersPipelineRepository
from ..service.create_pipeline_service import CreatePipelineService
from ..ui.pipeline_handler import PipelineHandler


class Container(containers.DeclarativeContainer):
    # 1. Repositories
    pipeline_repository = providers.Factory(DiffusersPipelineRepository)

    # 2. Services
    create_pipeline_service = providers.Factory(
        CreatePipelineService,
        pipeline_repo=pipeline_repository,
    )

    # 3. Handler
    pipeline_handler = providers.Factory(
        PipelineHandler,
        create_pipeline_service=create_pipeline_service,
    )
