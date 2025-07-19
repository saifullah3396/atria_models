import functools
import inspect

from hydra_zen import builds
from omegaconf import OmegaConf


def setup_model_config(attr_name="config"):
    def decorator(cls):
        original_init = cls.__init__

        @functools.wraps(original_init)
        def wrapped_init(self, *args, **kwargs):
            if type(self) is cls:
                sig = inspect.signature(original_init)
                bound = sig.bind(self, *args, **kwargs)
                bound.apply_defaults()

                # Build config dict (excluding 'self')
                config = {k: v for k, v in bound.arguments.items() if k != "self"}

                # flatten all kwargs
                if "model_kwargs" in config:
                    config.update(config["model_kwargs"])
                    del config["model_kwargs"]

                # Convert AtriaModel instances to their config
                setattr(
                    self,
                    attr_name,
                    OmegaConf.create(
                        builds(cls, populate_full_signature=True, **config)
                    ),
                )

            # Call the original __init__
            original_init(self, *args, **kwargs)

        cls.__init__ = wrapped_init
        return cls

    return decorator


def setup_model_pipeline_config(attr_name="config"):
    def decorator(cls):
        original_init = cls.__init__

        @functools.wraps(original_init)
        def wrapped_init(self, *args, **kwargs):
            from atria_models.core.atria_model import AtriaModel

            if type(self) is cls:
                sig = inspect.signature(original_init)
                bound = sig.bind(self, *args, **kwargs)
                bound.apply_defaults()

                config = {}
                for k, v in bound.arguments.items():
                    if k == "self":
                        continue
                    if k == "model":
                        if isinstance(v, dict) and all(
                            isinstance(item, AtriaModel) for item in v.values()
                        ):
                            config[k] = {k: v.config for k, v in v.items()}
                        elif isinstance(v, AtriaModel):
                            config[k] = v.config
                    elif k == "runtime_transforms":
                        config[k] = {
                            "train": v.train.config if v.train else None,
                            "evaluation": v.evaluation.config if v.evaluation else None,
                        }
                    else:
                        config[k] = v

                # Merge into existing config instead of overwriting
                setattr(
                    self,
                    attr_name,
                    OmegaConf.create(
                        builds(cls, populate_full_signature=True, **config)
                    ),
                )

            # Call the original __init__
            original_init(self, *args, **kwargs)

        cls.__init__ = wrapped_init
        return cls

    return decorator
