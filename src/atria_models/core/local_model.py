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

from atria_core.logger import get_logger
from torch.nn import Module

from atria_models.core.atria_model import AtriaModel

logger = get_logger(__name__)


class LocalModel(AtriaModel):
    """
    LocalModel Class

    This class is a specific implementation of the `AtriaModel` for managing local PyTorch models.
    It provides functionality for building models, configuring batch normalization layers, and
    managing frozen layers.

    Attributes:
        model_class (Type[Module]): The class of the PyTorch model to be instantiated.
        convert_bn_to_gn (bool): Flag indicating whether to convert batch normalization layers to group normalization.
        is_frozen (bool): Flag indicating whether the model is frozen.
        frozen_keys_patterns (Optional[List[str]]): List of patterns for keys to be frozen.
        unfrozen_keys_patterns (Optional[List[str]]): List of patterns for keys to be unfrozen.
        model_kwargs (dict): Additional keyword arguments for the model class.
    """

    def __init__(
        self,
        model_class: type[Module],
        convert_bn_to_gn: bool = False,
        is_frozen: bool = False,
        frozen_keys_patterns: list[str] | None = None,
        unfrozen_keys_patterns: list[str] | None = None,
        pretrained_checkpoint: str | None = None,
        **model_kwargs,
    ):
        """
        Initialize the LocalModel instance.

        Args:
            model_class (Type[Module]): The class of the PyTorch model to be instantiated.
            convert_bn_to_gn (bool, optional): Flag indicating whether to convert batch normalization layers to group normalization. Defaults to False.
            is_frozen (bool, optional): Flag indicating whether the model is frozen. Defaults to False.
            frozen_keys_patterns (Optional[List[str]], optional): List of patterns for keys to be frozen. Defaults to None.
            unfrozen_keys_patterns (Optional[List[str]], optional): List of patterns for keys to be unfrozen. Defaults to None.
            **model_kwargs: Additional keyword arguments for the model class.
        """
        self._model_class = model_class

        super().__init__(
            convert_bn_to_gn=convert_bn_to_gn,
            is_frozen=is_frozen,
            frozen_keys_patterns=frozen_keys_patterns,
            unfrozen_keys_patterns=unfrozen_keys_patterns,
            pretrained_checkpoint=pretrained_checkpoint,
            **model_kwargs,
        )

    @property
    def model_name(self) -> str:
        """
        Get the name of the `timm` model.

        Returns:
            str: The name of the `timm` model.
        """
        return self._model_class.__name__

    def _build(self, **kwargs) -> Module:
        if "pretrained" in kwargs:
            kwargs.pop("pretrained")
        return self._model_class(**kwargs)
