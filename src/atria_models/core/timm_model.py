"""
Timm Model Builder Module

This module defines the `TimmModel` class, which provides functionality
for constructing models from the `timm` library. It supports tasks such as image
classification and other tasks supported by `timm`.

Classes:
    - TimmModel: A model constructor for `timm` models.

Dependencies:
    - timm: For creating models from the `timm` library.
    - rich.pretty: For pretty-printing configurations.
    - torch: For PyTorch operations.
    - atria_core.logger.logger: For logging utilities.
    - atria_models.core.atria_model: For the base model class `AtriaModel`.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

import timm
from atria_core.logger.logger import get_logger
from torch.nn import Module

from atria_models.core.atria_model import AtriaModel
from atria_models.registry import MODEL

logger = get_logger(__name__)


@MODEL.register(
    "timm",
    model_name_pattern="${.model_name}",  # take the model name from the relative '_here_' config
)
class TimmModel(AtriaModel):
    """
    A model constructor for `timm` models.

    This class provides functionality for creating models from the `timm` library.
    It supports configurations such as pretraining, freezing layers, and converting
    batch normalization layers to group normalization layers.
    """

    def _build(self, *, num_labels: int | None = None, **kwargs) -> Module:
        """
        Build the `timm` model.

        This method constructs the model using the `timm.create_model` function
        with the specified configuration.

        Args:
            num_labels (int): The number of output labels for the model.
            **kwargs: Additional keyword arguments for the `timm.create_model` function.

        Returns:
            Module: The constructed `timm` model.
        """
        build_kwargs = {"model_name": self._model_name, **kwargs}
        if num_labels is not None:
            build_kwargs["num_classes"] = num_labels
        logger.info(
            "Setting up model %s(%s)",
            self.__class__.__name__,
            ", ".join(f"{k}={v!r}" for k, v in build_kwargs.items()),
        )
        return timm.create_model(**build_kwargs)
