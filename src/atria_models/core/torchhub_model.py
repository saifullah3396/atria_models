"""
TorchHubModel Model Builder Module

This module defines the `TorchHubModel` class, which provides functionality
for constructing models from the TorchHubModel library. It supports tasks such as image
classification and other tasks supported by TorchHubModel.

Classes:
    - TorchHubModel: A model constructor for TorchHubModel models.

Dependencies:
    - hydra_zen: For configuration management.
    - torch: For PyTorch operations and TorchHubModel hub.
    - atria_core.logger: For logging utilities.
    - atria_models.tasks: For defining model tasks.
    - atria_models.utilities.nn_modules: For neural network module utilities.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

import os
from typing import TYPE_CHECKING

from atria_core.logger.logger import get_logger
from rich.pretty import pretty_repr

from atria_models.core.atria_model import AtriaModel, AtriaModelConfig
from atria_models.registry import MODEL
from atria_models.utilities.nn_modules import (
    _get_last_module,
    _replace_module_with_name,
)

if TYPE_CHECKING:
    from torch.nn import Module

logger = get_logger(__name__)


class TorchHubConfig(AtriaModelConfig):
    hub_name: str = "???"


@MODEL.register("torchhub")
class TorchHubModel(AtriaModel):
    __config_cls__ = TorchHubConfig

    def _build(self, *, num_labels: int | None = None, **kwargs) -> "Module":
        """
        Constructs the TorchHubModel model.

        Returns:
            Module: The constructed TorchHubModel model.
        """
        import torch
        from torch.nn import Module

        logger.info(
            f"Initializing {self.__class__.__name__}/{self.config.model_name} with the following config:"
            f"\n{pretty_repr(kwargs, expand_all=True)}"
        )
        os.environ["TORCH_HOME"] = self.config.model_cache_dir
        self.config: TorchHubConfig
        model: Module = torch.hub.load(
            "pytorch/vision:v0.10.0", self.config.hub_name, verbose=False, **kwargs
        )
        if num_labels is not None:
            from torch.nn import Linear

            name, module = _get_last_module(model)
            _replace_module_with_name(
                model, name, Linear(module.in_features, num_labels)
            )
        else:
            logger.warning(
                "No 'num_labels' in 'model_initialization_kwargs' provided. "
                "Classification head will not be replaced."
            )

        return model
