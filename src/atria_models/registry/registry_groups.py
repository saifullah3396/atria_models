from atria_registry import RegistryGroup

from atria_models.utilities.config import (
    setup_model_config,
    setup_model_pipeline_config,
)


class ModelRegistryGroup(RegistryGroup):
    """
    A specialized registry group for managing models.

    This class provides additional methods for registering and managing models
    within the registry system.
    """

    def register(  # type: ignore[override]
        self,
        model_name: str,
        is_torch_model: bool = False,
        model_name_pattern: str | None = None,
        **kwargs,
    ) -> callable:
        """
        Decorator for registering a model.

        Args:
            model_name (str): The name of the model.
            is_nn_model (bool): Whether the model is a PyTorch neural network model. Defaults to False.
            model_name_pattern (str, optional): A pattern for the model name. Defaults to None.
            **kwargs: Additional keyword arguments for the registration.

        Returns:
            function: A decorator function for registering the model.
        """

        from atria_models.core.atria_model import AtriaModel

        def decorator(decorated_class):
            module_name = model_name
            if issubclass(decorated_class, AtriaModel):
                self.register_modules(
                    module_paths=decorated_class,
                    module_names=module_name,
                    zen_meta={
                        "name": (
                            model_name + "/" + model_name_pattern
                            if model_name_pattern
                            else model_name
                        )
                    },
                    **kwargs,
                )
            elif is_torch_model:
                from torch import nn

                from atria_models.core.local_model import LocalModel

                assert issubclass(decorated_class, nn.Module), (
                    f"Only torch.nn.Module or AtriaModel can be registered. {decorated_class} is not a subclass of torch.nn.Module or AtriaModel."
                )
                self.register_modules(
                    module_paths=LocalModel,
                    module_names=module_name,
                    model_class=decorated_class,
                    zen_meta={"name": module_name},
                    **kwargs,
                )
            else:
                raise TypeError(
                    f"Only AtriaModel or torch.nn.Module can be registered. {decorated_class} is not a subclass of AtriaModel or torch.nn.Module."
                )
            return setup_model_config()(decorated_class)

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
                return setup_model_pipeline_config()(decorated_class)
            else:
                module_name = name
                self.register_modules(
                    module_paths=decorated_class, module_names=module_name, **kwargs
                )
                return setup_model_pipeline_config()(decorated_class)

        return decorator
