"""
Atria Model Base Class Module

This module defines the `AtriaModel` class, which serves as a base class for all models
in the Atria framework. It provides functionality for building, validating, and configuring
PyTorch models, including support for freezing layers and converting BatchNorm to GroupNorm.

Classes:
    - AtriaModel: Base class for all Atria models.

Dependencies:
    - torch: For PyTorch operations.
    - atria_core.logger: For logging utilities.
    - atria_models.utilities.nn_modules: For neural network module utilities.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

from abc import abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, Any

from atria_core.logger.logger import get_logger
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from torch import nn

logger = get_logger(__name__)


class AtriaModelConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    model_name: str = "???"
    convert_bn_to_gn: bool = False
    is_frozen: bool = False
    frozen_keys_patterns: list[str] | None = None
    unfrozen_keys_patterns: list[str] | None = None
    pretrained_checkpoint: str | None = None
    model_cache_dir: str | None = None

    @property
    def model_kwargs(self) -> dict[str, Any]:
        return self.model_extra


class ModelConfigMixin:
    __config_cls__: type[AtriaModelConfig]
    __exclude_fields__: set[str] = set()

    def __init__(self, **kwargs):
        config_cls = getattr(self.__class__, "__config_cls__", None)
        assert issubclass(config_cls, AtriaModelConfig), (
            f"{self.__class__.__name__} must define a __config_cls__ attribute "
            "that is a subclass of AtriaModelConfig."
        )
        self._config = config_cls(**kwargs)
        super().__init__()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Validate presence of Config at class definition time
        if not hasattr(cls, "__config_cls__"):
            raise TypeError(
                f"{cls.__name__} must define a nested `__config_cls__` class."
            )

        if not issubclass(cls.__config_cls__, AtriaModelConfig):
            raise TypeError(
                f"{cls.__name__}.Config must subclass pydantic.AtriaModelConfig. Got {cls.__config_cls__} instead."
            )

    def prepare_build_config(self):
        from hydra_zen import builds
        from omegaconf import OmegaConf

        if self.__config_cls__ is None:
            raise TypeError(
                f"{self.__class__.__name__} must define a __config_cls__ attribute."
            )
        init_fields = {
            k: getattr(self._config, k) for k in self._config.__class__.model_fields
        }
        for key in self.__exclude_fields__:
            init_fields.pop(key)
        return OmegaConf.to_container(
            OmegaConf.create(
                builds(self.__class__, populate_full_signature=True, **init_fields)
            )
        )

    @cached_property
    def config(self) -> AtriaModelConfig:
        return self._config

    @cached_property
    def build_config(self) -> AtriaModelConfig:
        return self.prepare_build_config()

    @cached_property
    def config_hash(self) -> str:
        """
        Hash of the dataset configuration for versioning.

        Returns:
            8-character hash string based on configuration content
        """
        import hashlib
        import json

        return hashlib.sha256(
            json.dumps(self.build_config, sort_keys=True).encode()
        ).hexdigest()[:8]


class AtriaModel(ModelConfigMixin):
    __config_cls__ = AtriaModelConfig
    __exclude_fields__ = {"model_cache_dir"}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.config.model_cache_dir is None:
            from atria_core.constants import _DEFAULT_ATRIA_MODELS_CACHE_DIR

            self.config.model_cache_dir = str(_DEFAULT_ATRIA_MODELS_CACHE_DIR)

    @property
    def model_name(self) -> str:
        """
        Get the name of the model.

        Returns:
            str: The name of the model class.
        """
        return self.config.model_name

    @property
    def is_frozen(self) -> bool:
        """
        Check if the model is frozen.

        Returns:
            bool: True if the model is frozen, False otherwise.
        """
        return self.config.is_frozen

    def _validate_model(self, model: Any) -> nn.Module:
        """
        Validate the model after building it.

        Args:
            model (Any): The model to validate.

        Returns:
            torch.Module: The validated PyTorch model.

        Raises:
            ValueError: If the model is not a valid PyTorch module.
        """

        from torch.nn import Module

        if not isinstance(model, Module):
            raise ValueError(f"Model is not a valid PyTorch module. Got {type(model)}")
        return model

    def _configure_batch_norm_layers(self, model: nn.Module) -> None:
        """
        Configure BatchNorm layers in the model.

        Converts BatchNorm layers to GroupNorm layers if `convert_bn_to_gn` is True.
        """

        from atria_models.utilities.nn_modules import _batch_norm_to_group_norm

        if self.config.convert_bn_to_gn:
            logger.warning(
                "Converting BatchNorm layers to GroupNorm layers in the model. "
                "If this is not intended, set convert_bn_to_gn=False."
            )
            _batch_norm_to_group_norm(model)

    def _configure_model_frozen_layers(self, model: nn.Module) -> None:
        """
        Configure frozen layers in the model.

        Freezes the entire model if `is_frozen` is True. Otherwise, applies freeze and unfreeze
        patterns to specific layers based on `frozen_keys_patterns` and `unfrozen_keys_patterns`.
        """
        import json

        from atria_models.utilities.nn_modules import _freeze_layers_with_key_pattern

        if self.config.is_frozen:
            logger.warning(
                "Freezing the model. If this is not intended, set is_frozen=False in its config."
            )
            model.requires_grad_(False)
        else:
            if self.config.frozen_keys_patterns or self.config.unfrozen_keys_patterns:
                logger.info(
                    f"Applying freeze patterns: {self.config.frozen_keys_patterns}"
                )
                logger.info(
                    f"Applying unfreeze patterns: {self.config.unfrozen_keys_patterns}"
                )
                trainable_params = _freeze_layers_with_key_pattern(
                    model=model,
                    frozen_keys_patterns=self.config.frozen_keys_patterns,
                    unfrozen_keys_patterns=self.config.unfrozen_keys_patterns,
                )
                logger.info(
                    f"Trainable parameters: {json.dumps(trainable_params, indent=2)}"
                )

    def build(self, **kwargs) -> nn.Module:
        """
        Build the model by calling the abstract method `_build`.

        Args:
            **kwargs: Additional keyword arguments for model building.

        Returns:
            torch.Module: The built PyTorch model.
        """

        from atria_models.utilities.checkpoints import _load_checkpoint_from_path_or_url

        model = self._validate_model(self._build(**kwargs, **self.config.model_kwargs))
        if self.config.pretrained_checkpoint is not None:
            checkpoint = _load_checkpoint_from_path_or_url(
                self.config.pretrained_checkpoint
            )
            missing_keys, unexpected_keys = model.load_state_dict(
                checkpoint, strict=False
            )
            if missing_keys or unexpected_keys:
                logger.warning(
                    "Model loaded with missing or unexpected keys:\n"
                    f"Missing keys: {missing_keys}\n"
                    f"Unexpected keys: {unexpected_keys}"
                )
        self._configure_batch_norm_layers(model)
        self._configure_model_frozen_layers(model)
        return model

    @abstractmethod
    def _build(self, **kwargs) -> nn.Module:
        """
        Abstract method to initialize the model. Must be implemented by subclasses.

        Args:
            **kwargs: Additional keyword arguments for model initialization.

        Returns:
            torch.Module: The initialized PyTorch model.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError(
            "Subclasses must implement the _build method to initialize the model."
        )
