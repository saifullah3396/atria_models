"""
MMDetection Model Builder Module

This module defines the `MMDetModel` class, which provides functionality
for constructing models from the MMDetection library. It supports tasks such as
object detection and other MMDetection-supported tasks.

Classes:
    - MMDetModel: A model constructor for MMDetection models.

Dependencies:
    - importlib: For dynamic imports and module resolution.
    - pathlib.Path: For handling file system paths.
    - typing: For type annotations.
    - torch: For PyTorch operations.
    - atria_core.logger.logger: For logging utilities.
    - atria_core.utilities.imports: For utility functions related to imports.
    - atria_models.core.atria_model: For the base model class `AtriaModel`.
    - mmdet.registry: For MMDetection model registry.
    - mmdet.utils: For registering MMDetection modules.
    - mmengine.config: For loading MMDetection configuration files.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

import importlib
from pathlib import Path
from typing import TYPE_CHECKING

from atria_core.logger.logger import get_logger
from atria_core.utilities.imports import _get_package_base_path

from atria_models.core.atria_model import AtriaModel, AtriaModelConfig
from atria_models.registry import MODEL

if TYPE_CHECKING:
    from mmdet.models.detectors import BaseDetector

logger = get_logger(__name__)

_DEFAULT_MMDET_CONFIG_PATH = Path(_get_package_base_path("atria_models")) / "conf_mmdet"


class MMdetModelConfig(AtriaModelConfig):
    mmdet_name: str = "???"  # Placeholder for the model name
    model_search_paths: list[str] | None = None


@MODEL.register("mmdet")
class MMDetModel(AtriaModel):
    __config_cls__ = MMdetModelConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.config: MMdetModelConfig
        if self.config.model_search_paths is not None:
            self._resolved_model_search_paths = []
            for path in self.config.model_search_paths:
                if path.startswith("pkg://"):
                    filtered_path = path.replace("pkg://", "").split("/")
                    module_name = filtered_path[0]
                    if len(filtered_path) > 1:
                        path = "/".join(filtered_path[1:])
                    else:
                        path = ""
                    spec = importlib.util.find_spec(module_name)
                    assert spec is not None, f"Path {path} not found."
                    self._resolved_model_search_paths.append(
                        Path(spec.origin).parent / path
                    )
                else:
                    self._resolved_model_search_paths.append(
                        Path(path).resolve(strict=True)
                    )
        else:
            self._resolved_model_search_paths = [_DEFAULT_MMDET_CONFIG_PATH]

    def _build(self, **kwargs) -> "BaseDetector":
        """
        Constructs the MMDetection model based on the provided configuration.

        This method searches for the model configuration in the specified paths,
        loads the configuration, and initializes the model.

        Returns:
            BaseDetector: The initialized MMDetection model.

        Raises:
            FileNotFoundError: If the model configuration file is not found in the search paths.
        """
        from mmdet.registry import MODELS
        from mmdet.utils import register_all_modules
        from mmengine.config import Config

        # Register all the modules from mmdet to the registry
        register_all_modules()

        # Find the model configuration in the search paths
        for path in self._resolved_model_search_paths:
            if (path / self.config.mmdet_name).exists():
                cfg = Config.fromfile(path / self.config.mmdet_name)
                for key, value in kwargs.items():
                    if hasattr(cfg.model, key):
                        setattr(cfg.model, key, value)
                return MODELS.build(cfg.model)

        raise FileNotFoundError(
            f"Model {self.config.mmdet_name} not found in search paths {[str(x) for x in self._resolved_model_search_paths]}"
        )
