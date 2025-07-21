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

from atria_core.constants import _DEFAULT_ATRIA_MODELS_CACHE_DIR
from atria_core.logger.logger import get_logger
from rich.pretty import pretty_repr

from atria_models.core.atria_model import AtriaModel
from atria_models.registry import MODEL
from atria_models.utilities.nn_modules import (
    _get_last_module,
    _replace_module_with_name,
)

if TYPE_CHECKING:
    from torch.nn import Module

logger = get_logger(__name__)


@MODEL.register(
    "torchhub",
    model_name_pattern="${.model_name}",  # take the model name from the relative '_here_' config
)
class TorchHubModel(AtriaModel):
    """
    A model builder class for constructing models from the torch hub library.

    Attributes:
        model_name (str): The name of the torch hub model to be constructed.
        num_labels (Optional[int]): The number of labels for the classification task.
        model_cache_dir (Optional[str]): Directory for caching torch hub models.
        pretrained (bool): Whether to use pretrained weights for the model.
        convert_bn_to_gn (bool): Whether to convert BatchNorm layers to GroupNorm.
        is_frozen (bool): Whether to freeze the model parameters.
        frozen_keys_patterns (Optional[List[str]]): Patterns for keys to freeze.
        unfrozen_keys_patterns (Optional[List[str]]): Patterns for keys to unfreeze.
        model_kwargs (dict): Additional keyword arguments for model initialization.
    """

    def __init__(
        self,
        model_name: str,
        model_cache_dir: str | None = None,
        convert_bn_to_gn: bool = False,
        is_frozen: bool = False,
        frozen_keys_patterns: list[str] | None = None,
        unfrozen_keys_patterns: list[str] | None = None,
        pretrained_checkpoint: str | None = None,
        **model_kwargs,
    ):
        """
        Initializes the TorchHubModel instance.

        Args:
            model_name (str): The name of the TorchHubModel model to be constructed.
            num_labels (Optional[int]): The number of labels for the classification task.
            model_cache_dir (Optional[str]): Directory for caching TorchHubModel models.
            pretrained (bool): Whether to use pretrained weights for the model.
            convert_bn_to_gn (bool): Whether to convert BatchNorm layers to GroupNorm.
            is_frozen (bool): Whether to freeze the model parameters.
            frozen_keys_patterns (Optional[List[str]]): Patterns for keys to freeze.
            unfrozen_keys_patterns (Optional[List[str]]): Patterns for keys to unfreeze.
            model_kwargs (dict): Additional keyword arguments for model initialization.
        """
        self._model_cache_dir = model_cache_dir or _DEFAULT_ATRIA_MODELS_CACHE_DIR

        super().__init__(
            model_name=model_name,
            convert_bn_to_gn=convert_bn_to_gn,
            is_frozen=is_frozen,
            frozen_keys_patterns=frozen_keys_patterns,
            unfrozen_keys_patterns=unfrozen_keys_patterns,
            pretrained_checkpoint=pretrained_checkpoint,
            **model_kwargs,
        )

    def _build(self, *, num_labels: int | None = None, **kwargs) -> "Module":
        """
        Constructs the TorchHubModel model.

        Returns:
            Module: The constructed TorchHubModel model.
        """
        import torch
        from torch.nn import Module

        logger.info(
            f"Initializing {self.__class__.__name__}/{self._model_name} with the following config:"
            f"\n{pretty_repr(kwargs, expand_all=True)}"
        )
        os.environ["TORCH_HOME"] = self._model_cache_dir
        model: Module = torch.hub.load(
            "pytorch/vision:v0.10.0", self._model_name, verbose=False, **kwargs
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
