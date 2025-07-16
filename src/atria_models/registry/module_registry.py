from atria_registry.module_registry import ModuleRegistry
from atria_registry.registry_group import RegistryGroup

from atria_models.registry.registry_groups import ModelRegistryGroup

_initialized = False


def init_registry():
    global _initialized
    if _initialized:
        return
    _initialized = True
    ModuleRegistry().add_registry_group(
        name="MODEL", registry_group=ModelRegistryGroup(name="model")
    )
    ModuleRegistry().add_registry_group(
        name="MODEL_PIPELINE", registry_group=RegistryGroup(name="model_pipeline")
    )
