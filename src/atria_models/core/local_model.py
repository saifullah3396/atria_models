"""
Local Model Module

This module defines the `LocalModel` class, which is a specific implementation of the `AtriaModel`
for managing local PyTorch models. It provides functionality for building models, configuring
batch normalization layers, and managing frozen layers.

Classes:
    - LocalModel: A class for managing local PyTorch models.

Dependencies:
    - torch: For PyTorch operations.
    - atria_core.logger: For logging utilities.
    - atria_models.core.atria_model: For the base `AtriaModel` class.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from collections.abc import Callable
from inspect import isclass
from typing import TYPE_CHECKING, Any

from atria_core.logger import get_logger
from pydantic import model_validator

from atria_models.core.atria_model import AtriaModel, AtriaModelConfig

if TYPE_CHECKING:
    from torch.nn import Module

logger = get_logger(__name__)


class LocalModelConfig(AtriaModelConfig):
    module: type | Callable  # The class of the model to be instantiated

    @model_validator(mode="before")
    @classmethod
    def validate_model_name(cls, values: dict[str, Any]) -> dict[str, Any]:
        from atria_core.utilities.strings import _convert_to_snake_case

        if "model_name" not in values:
            values["model_name"] = _convert_to_snake_case(values["module"].__name__)
        return values


class LocalModel(AtriaModel):
    __config_cls__ = LocalModelConfig

    def _build(self, **kwargs) -> "Module":
        import inspect

        self.config: LocalModelConfig

        # Get the init signature of the model class
        signature = (
            inspect.signature(self.config.module.__init__)
            if isclass(self.config.module)
            else inspect.signature(self.config.module)
        )
        valid_params = signature.parameters

        # Filter kwargs to only include valid init parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

        logger.info(
            "Setting up model %s with kwargs=(%s)",
            self.config.module,
            ", ".join(f"{k}={v!r}" for k, v in filtered_kwargs.items()),
        )

        return self.config.module(**filtered_kwargs)
