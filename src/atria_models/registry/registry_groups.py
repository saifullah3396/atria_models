from atria_registry import RegistryGroup


class ModelRegistryGroup(RegistryGroup):
    """
    A specialized registry group for managing models.

    This class provides additional methods for registering and managing models
    within the registry system.
    """

    def register(  # type: ignore[override]
        self, model_name: str, model_name_pattern: str | None = None, **kwargs
    ) -> callable:
        """
        Decorator for registering a model.

        Args:
            model_name (str): The name of the model.
            model_name_pattern (str, optional): A pattern for the model name. Defaults to None.
            **kwargs: Additional keyword arguments for the registration.

        Returns:
            function: A decorator function for registering the model.
        """
        import torch

        from atria_models.core.atria_model import AtriaModel
        from atria_models.core.local_model import LocalModel

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
            else:
                assert issubclass(decorated_class, torch.nn.Module), (
                    f"Only torch.nn.Module or AtriaModel can be registered. {decorated_class} is not a subclass of torch.nn.Module or AtriaModel."
                )
                self.register_modules(
                    module_paths=LocalModel,
                    module_names=module_name,
                    model_class=decorated_class,
                    zen_meta={"name": module_name},
                    **kwargs,
                )
            return decorated_class

        return decorator
