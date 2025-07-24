from __future__ import annotations

from typing import TYPE_CHECKING

from atria_registry import RegistryGroup

if TYPE_CHECKING:
    from atria_models.core.atria_model import AtriaModelConfig


class ModelRegistryGroup(RegistryGroup):
    """
    A specialized registry group for managing models.

    This class provides additional methods for registering and managing models
    within the registry system.
    """

    def register(
        self, name: str, configs: list[AtriaModelConfig] | None = None, **kwargs
    ):
        """
        Decorator for registering a module with configurations.

        Args:
            name (str): The name of the module.
            **kwargs: Additional keyword arguments for the registration.

        Returns:
            function: A decorator function for registering the module with configurations.
        """

        def decorator(decorated_class):
            from atria_models.core.atria_model import AtriaModel, AtriaModelConfig
            from atria_models.core.local_model import LocalModel

            module_path = decorated_class
            module_name = name
            config_cls = None
            config_cls_kwargs = {}

            if not issubclass(decorated_class, AtriaModel):
                from torch import nn

                if issubclass(decorated_class, nn.Module):
                    module_path = LocalModel
                    module_name = f"atria/{module_name}"
                    config_cls = LocalModel.__config_cls__
                    config_cls_kwargs = {"model_class": decorated_class}
                else:
                    raise TypeError(
                        f"Only AtriaModel or torch.nn.Module can be registered. {decorated_class} is not a subclass of AtriaModel or torch.nn.Module."
                    )
            else:
                config_cls = decorated_class.__config_cls__

            if configs is not None:
                assert isinstance(configs, list) and all(
                    isinstance(config, AtriaModelConfig) for config in configs
                ), (
                    f"Expected configs to be a list of AtriaModelConfig, got {type(configs)} instead."
                )
                for config in configs:
                    self.register_modules(
                        module_paths=module_path,
                        module_names=module_name + "/" + config.name,
                        **{
                            k: getattr(config, k) for k in config.__class__.model_fields
                        },
                        **kwargs,
                    )
                return decorated_class
            else:
                config = config_cls(**config_cls_kwargs)
                self.register_modules(
                    module_paths=module_path,
                    module_names=module_name,
                    **{k: getattr(config, k) for k in config.__class__.model_fields},
                    **kwargs,
                )
                return decorated_class

        return decorator


class ModelPipelineRegistryGroup(RegistryGroup):
    """
    A specialized registry group for managing model pipeline.

    This class provides additional methods for registering and managing pipeline
    within the registry system.
    """

    def register(self, name: str, **kwargs):
        """
        Decorator for registering a module with configurations.

        Args:
            name (str): The name of the module.
            **kwargs: Additional keyword arguments for the registration.

        Returns:
            function: A decorator function for registering the module with configurations.
        """

        def decorator(decorated_class):
            if hasattr(decorated_class, "_REGISTRY_CONFIGS"):
                configs = decorated_class._REGISTRY_CONFIGS
                assert isinstance(configs, dict), (
                    f"Expected _REGISTRY_CONFIGS on {decorated_class.__name__} to be a dict, "
                    f"but got {type(configs).__name__} instead."
                )
                assert configs, (
                    f"{decorated_class.__name__} must provide at least one configuration in _REGISTRY_CONFIGS."
                )
                for key, config in configs.items():
                    assert isinstance(config, dict), (
                        f"Configuration {config} must be a dict."
                    )
                    module_name = name
                    self.register_modules(
                        module_paths=decorated_class,
                        module_names=module_name + "/" + key,
                        **config,
                        **kwargs,
                    )
                return decorated_class
            else:
                module_name = name
                self.register_modules(
                    module_paths=decorated_class, module_names=module_name, **kwargs
                )
                return decorated_class

        return decorator
