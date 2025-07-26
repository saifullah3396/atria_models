from __future__ import annotations

from inspect import isclass
from typing import TYPE_CHECKING

from atria_registry import RegistryGroup

if TYPE_CHECKING:
    from atria_models.core.atria_model import AtriaModelConfig
    from atria_models.pipelines.atria_model_pipeline import AtriaModelPipelineConfig


class ModelRegistryGroup(RegistryGroup):
    """
    A specialized registry group for managing models.

    This class provides additional methods for registering and managing models
    within the registry system.
    """

    def register(
        self,
        name: str,
        configs: list[AtriaModelConfig] | None = None,
        builds_to_file_store: bool = True,
        **kwargs,
    ):
        """
        Decorator for registering a module with configurations.

        Args:
            name (str): The name of the module.
            **kwargs: Additional keyword arguments for the registration.

        Returns:
            function: A decorator function for registering the module with configurations.
        """

        if builds_to_file_store and not self._file_store_build_enabled:

            def noop_(module):
                return module

            return noop_

        # get spec params
        provider = kwargs.pop("provider", None)
        is_global_package = kwargs.pop("is_global_package", False)
        registers_target = kwargs.pop("registers_target", True)
        defaults = kwargs.pop("defaults", None)
        assert defaults is None, "Dataset registry does not support defaults."

        def decorator(module):
            from atria_registry.module_spec import ModuleSpec

            from atria_models.core.atria_model import AtriaModel, AtriaModelConfig
            from atria_models.core.local_model import LocalModel

            # build the module spec
            module_spec = ModuleSpec(
                module=module,
                name=name,
                group=self.name,
                provider=provider or self._default_provider,
                is_global_package=is_global_package,
                registers_target=registers_target,
                defaults=defaults,
            )

            if isclass(module):
                if not issubclass(module, AtriaModel):
                    from torch import nn

                    assert issubclass(module, nn.Module), (
                        f"Only AtriaModel or torch.nn.Module can be registered. {module} is not a subclass of AtriaModel or torch.nn.Module."
                    )

                    # this is a parameter of the LocalModelConfig not to be confused with the module parameter for the
                    # registry, we pass it to the LocalModelConfig to instantiate the model
                    config = LocalModel.__config_cls__(
                        module=module_spec.module
                    )  # we first pass it to config then replace it!!
                    module_spec.module = LocalModel
                    module_spec.model_extra.update(
                        {k: getattr(config, k) for k in config.__class__.model_fields}
                    )
                else:
                    config = module.__config_cls__()
                    module_spec.model_extra.update(
                        {k: getattr(config, k) for k in config.__class__.model_fields}
                    )

                if configs is not None:
                    import copy

                    assert isinstance(configs, list) and all(
                        isinstance(config, AtriaModelConfig) for config in configs
                    ), (
                        f"Expected configs to be a list of AtriaModelConfig, got {type(configs)} instead."
                    )

                    for config in configs:
                        config.model_name = name
                        config_module_spec = copy.deepcopy(module_spec)
                        config_module_spec.name = (
                            config.model_name + "/" + config.config_name
                        )
                        config_module_spec.model_extra.update(
                            {
                                k: getattr(config, k)
                                for k in config.__class__.model_fields
                            }
                        )
                        config_module_spec.model_extra.update({**kwargs})
                        self.register_module(config_module_spec)
                    return module
                module_spec.model_extra.update({**kwargs})
                self.register_module(module_spec)
            else:
                # we initialize all callable functions as LocalModel which will return
                # the model class on call
                config = LocalModel.__config_cls__(
                    module=module_spec.module
                )  # we first pass it to config then replace it!!
                module_spec.module = LocalModel
                module_spec.model_extra.update(
                    {k: getattr(config, k) for k in config.__class__.model_fields}
                )
                module_spec.model_extra.update({**kwargs})
                self.register_module(module_spec)
            return module

        return decorator


class ModelPipelineRegistryGroup(RegistryGroup):
    """
    A specialized registry group for managing models.

    This class provides additional methods for registering and managing models
    within the registry system.
    """

    def register(
        self,
        name: str,
        configs: list[AtriaModelPipelineConfig] | None = None,
        builds_to_file_store: bool = True,
        **kwargs,
    ):
        """
        Decorator for registering a module with configurations.

        Args:
            name (str): The name of the module.
            **kwargs: Additional keyword arguments for the registration.

        Returns:
            function: A decorator function for registering the module with configurations.
        """
        if builds_to_file_store and not self._file_store_build_enabled:

            def noop_(module):
                return module

            return noop_

        # get spec params
        provider = kwargs.pop("provider", None)
        is_global_package = kwargs.pop("is_global_package", False)
        registers_target = kwargs.pop("registers_target", True)
        defaults = kwargs.pop("defaults", None)

        def decorator(module):
            from atria_registry.module_spec import ModuleSpec

            from atria_models.pipelines.atria_model_pipeline import (
                AtriaModelPipelineConfig,
            )

            # build the module spec
            module_spec = ModuleSpec(
                module=module,
                name=name,
                group=self.name,
                provider=provider or self._default_provider,
                is_global_package=is_global_package,
                registers_target=registers_target,
                defaults=defaults,
            )

            if configs is not None:
                import copy

                assert isinstance(configs, list) and all(
                    isinstance(config, AtriaModelPipelineConfig) for config in configs
                ), (
                    f"Expected configs to be a list of RegistryConfig, got {type(configs)} instead."
                )
                for config in configs:
                    config_module_spec = copy.deepcopy(module_spec)
                    config_defaults = config.model_extra.pop("defaults", None)
                    if config_defaults is not None:
                        config_module_spec.defaults = config_defaults
                    config_module_spec.name = (
                        config_module_spec.name + "/" + config.name
                    )
                    config_module_spec.model_extra.update(
                        {**config.model_extra, **kwargs}
                    )
                    self.register_module(config_module_spec)
                return module
            module_spec.model_extra.update({**kwargs})
            self.register_module(module_spec)
            return module

        return decorator
