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

from typing import TYPE_CHECKING, Any

from atria_core.logger import get_logger
from pydantic import model_validator

from atria_models.core.atria_model import AtriaModel, AtriaModelConfig

if TYPE_CHECKING:
    from torch.nn import Module

logger = get_logger(__name__)


class LocalModelConfig(AtriaModelConfig):
    model_class: type  # The class of the model to be instantiated

    @model_validator(mode="before")
    @classmethod
    def validate_model_name(cls, values: dict[str, Any]) -> dict[str, Any]:
        if "model_name" not in values:
            values["model_name"] = values["model_class"].__name__
        return values


class LocalModel(AtriaModel):
    __config_cls__ = LocalModelConfig

    def _build(self, **kwargs) -> "Module":
        import inspect

        self.config: LocalModelConfig

        # Get the init signature of the model class
        signature = inspect.signature(self.config.model_class.__init__)
        valid_params = signature.parameters

        # Filter kwargs to only include valid init parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

        logger.info(
            "Setting up model %s with kwargs=(%s)",
            self.config.model_class,
            ", ".join(f"{k}={v!r}" for k, v in filtered_kwargs.items()),
        )

        return self.config.model_class(**filtered_kwargs)
