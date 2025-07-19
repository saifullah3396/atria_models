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

import importlib

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
    """

    def _build(self, **kwargs) -> Module:
        if "pretrained" in kwargs:
            kwargs.pop("pretrained")

        # Split the full path to get module and class name
        module_path, class_name = self.model_name.rsplit(".", 1)
        model_module = importlib.import_module(module_path)
        model_class = getattr(model_module, class_name)
        return model_class(**kwargs)
