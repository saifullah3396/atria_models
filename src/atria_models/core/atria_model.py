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

import json
from abc import abstractmethod
from typing import Any

from atria_core.logger.logger import get_logger
from torch.nn import Module

from atria_models.utilities.checkpoints import CheckpointManager
from atria_models.utilities.nn_modules import (
    _batch_norm_to_group_norm,
    _freeze_layers_with_key_pattern,
)

logger = get_logger(__name__)


class AtriaModel(Module):
    """
    Base class for all Atria models.

    This class provides functionality for building, validating, and configuring PyTorch models.
    It includes support for freezing layers and converting BatchNorm layers to GroupNorm layers.

    Attributes:
        model_name (str): The name of the model class.
        convert_bn_to_gn (bool): Whether to convert BatchNorm layers to GroupNorm layers.
        is_frozen (bool): Whether the model is frozen.
        frozen_keys_patterns (Optional[List[str]]): Patterns for freezing specific layers.
        unfrozen_keys_patterns (Optional[List[str]]): Patterns for unfreezing specific layers.
        model_kwargs (dict): Additional keyword arguments for model configuration.
    """

    def __init__(
        self,
        model_name: str,
        convert_bn_to_gn: bool = False,
        is_frozen: bool = False,
        frozen_keys_patterns: list[str] | None = None,
        unfrozen_keys_patterns: list[str] | None = None,
        pretrained_checkpoint: str | None = None,
        **model_kwargs,
    ):
        """
        Initialize the AtriaModel instance.

        Args:
            convert_bn_to_gn (bool): Whether to convert BatchNorm layers to GroupNorm layers.
            is_frozen (bool): Whether the model is frozen.
            frozen_keys_patterns (Optional[List[str]]): Patterns for freezing specific layers.
            unfrozen_keys_patterns (Optional[List[str]]): Patterns for unfreezing specific layers.
            pretrained_checkpoint (Optional[str]): Path to a pretrained checkpoint.
            **model_kwargs: Additional keyword arguments for model configuration.
        """

        Module.__init__(self)
        self._model_name = model_name
        self._convert_bn_to_gn = convert_bn_to_gn
        self._is_frozen = is_frozen
        self._frozen_keys_patterns = frozen_keys_patterns
        self._unfrozen_keys_patterns = unfrozen_keys_patterns
        self._pretrained_checkpoint = pretrained_checkpoint
        self._model: Module = None
        self._model_kwargs = model_kwargs

    @property
    def model_name(self) -> str:
        """
        Get the name of the model.

        Returns:
            str: The name of the model class.
        """
        return self.model_name

    @property
    def is_frozen(self) -> bool:
        """
        Check if the model is frozen.

        Returns:
            bool: True if the model is frozen, False otherwise.
        """
        return self._is_frozen

    def _validate_model(self, model: Any) -> Module:
        """
        Validate the model after building it.

        Args:
            model (Any): The model to validate.

        Returns:
            torch.Module: The validated PyTorch model.

        Raises:
            ValueError: If the model is not a valid PyTorch module.
        """

        if not isinstance(model, Module):
            raise ValueError(
                f"Model is not a valid PyTorch module. Got {type(self._model)}"
            )
        return model

    def _configure_batch_norm_layers(self) -> None:
        """
        Configure BatchNorm layers in the model.

        Converts BatchNorm layers to GroupNorm layers if `convert_bn_to_gn` is True.
        """
        if self._convert_bn_to_gn:
            logger.warning(
                "Converting BatchNorm layers to GroupNorm layers in the model. "
                "If this is not intended, set convert_bn_to_gn=False."
            )
            _batch_norm_to_group_norm(self._model)

    def _configure_model_frozen_layers(self) -> None:
        """
        Configure frozen layers in the model.

        Freezes the entire model if `is_frozen` is True. Otherwise, applies freeze and unfreeze
        patterns to specific layers based on `frozen_keys_patterns` and `unfrozen_keys_patterns`.
        """
        if self._is_frozen:
            logger.warning(
                "Freezing the model. If this is not intended, set is_frozen=False in its config."
            )
            self._model.requires_grad_(False)
        else:
            if self._frozen_keys_patterns or self._unfrozen_keys_patterns:
                logger.info(f"Applying freeze patterns: {self._frozen_keys_patterns}")
                logger.info(
                    f"Applying unfreeze patterns: {self._unfrozen_keys_patterns}"
                )
                trainable_params = _freeze_layers_with_key_pattern(
                    model=self._model,
                    frozen_keys_patterns=self._frozen_keys_patterns,
                    unfrozen_keys_patterns=self._unfrozen_keys_patterns,
                )
                logger.info(
                    f"Trainable parameters: {json.dumps(trainable_params, indent=2)}"
                )

    def build(self, **kwargs) -> "AtriaModel":
        """
        Build the model by calling the abstract method `_build`.

        Args:
            **kwargs: Additional keyword arguments for model building.

        Returns:
            AtriaModel: The built Atria model instance.
        """
        self._model = self._validate_model(
            self._build(**{**self._model_kwargs, **kwargs})
        )
        if self._pretrained_checkpoint is not None:
            checkpoint = CheckpointManager._load_checkpoint_from_path_or_url(
                self._pretrained_checkpoint
            )
            CheckpointManager._apply_checkpoint_to_model(
                self._model, checkpoint, strict=False
            )
        self._configure_batch_norm_layers()
        self._configure_model_frozen_layers()
        return self

    @abstractmethod
    def _build(self, **kwargs) -> Module:
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

    def forward(self, *args, **kwargs) -> Any:
        """
        Forward pass through the model.

        Args:
            *args: Positional arguments for the model's forward method.
            **kwargs: Keyword arguments for the model's forward method.

        Returns:
            Any: The output of the model.
        """
        return self._model(*args, **kwargs)
