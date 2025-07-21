from atria_registry.module_registry import ModuleRegistry

from atria_models.registry.registry_groups import (
    ModelPipelineRegistryGroup,
    ModelRegistryGroup,
)

_initialized = False


def init_registry():
    global _initialized
    if _initialized:
        return
    _initialized = True
    ModuleRegistry().add_registry_group(
        name="MODEL",
        registry_group=ModelRegistryGroup(
            name="model", default_provider="atria_models"
        ),
    )
    ModuleRegistry().add_registry_group(
        name="MODEL_PIPELINE",
        registry_group=ModelPipelineRegistryGroup(
            name="model_pipeline", default_provider="atria_models"
        ),
    )
