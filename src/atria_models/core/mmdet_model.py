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

from atria_core.constants import _DEFAULT_ATRIA_MODELS_CACHE_DIR
from atria_core.logger.logger import get_logger
from atria_core.utilities.imports import _get_package_base_path

from atria_models.core.atria_model import AtriaModel
from atria_models.registry import MODEL

if TYPE_CHECKING:
    from mmdet.models.detectors import BaseDetector

logger = get_logger(__name__)

_DEFAULT_MMDET_CONFIG_PATH = Path(_get_package_base_path("atria_models")) / "conf_mmdet"


@MODEL.register(
    "mmdet",
    model_name_pattern="${.model_name}",  # take the model name from the relative '_here_' config
)
class MMDetModel(AtriaModel):
    """
    A model constructor for MMDetection models.

    This class provides functionality for constructing models from the MMDetection
    library. It supports tasks such as object detection and other MMDetection-supported tasks.

    Attributes:
        model_name (str): The name of the model to be constructed.
        model_search_paths (Optional[List[str]]): Paths to search for model configurations.
        num_labels (Optional[int]): The number of output labels for the model.
        model_cache_dir (Optional[str]): The directory where the model weights are stored.
        convert_bn_to_gn (bool): Whether to convert BatchNorm layers to GroupNorm layers.
        is_frozen (bool): Whether to freeze the model.
        frozen_keys_patterns (Optional[List[str]]): Patterns to freeze model layers.
        unfrozen_keys_patterns (Optional[List[str]]): Patterns to unfreeze model layers.
    """

    def __init__(
        self,
        model_name: str,
        model_search_paths: list[str] | None = None,
        model_cache_dir: str | None = None,
        convert_bn_to_gn: bool = False,
        is_frozen: bool = False,
        frozen_keys_patterns: list[str] | None = None,
        unfrozen_keys_patterns: list[str] | None = None,
        pretrained_checkpoint: str | None = None,
        **model_kwargs,
    ):
        """
        Initialize the MMDetModel instance.

        Args:
            model_name (str): The name of the model to be constructed.
            model_search_paths (Optional[List[str]], optional): Paths to search for model configurations. Defaults to None.
            model_cache_dir (Optional[str], optional): The directory where the model weights are stored. Defaults to None.
            convert_bn_to_gn (bool, optional): Whether to convert BatchNorm layers to GroupNorm layers. Defaults to False.
            is_frozen (bool, optional): Whether to freeze the model. Defaults to False.
            frozen_keys_patterns (Optional[List[str]], optional): Patterns to freeze model layers. Defaults to None.
            unfrozen_keys_patterns (Optional[List[str]], optional): Patterns to unfreeze model layers. Defaults to None.
            **model_kwargs: Additional keyword arguments for the model.
        """
        self._model_cache_dir = model_cache_dir or _DEFAULT_ATRIA_MODELS_CACHE_DIR

        if model_search_paths is not None:
            self._resolved_model_search_paths = []
            for path in model_search_paths:
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

        super().__init__(
            model_name=model_name,
            convert_bn_to_gn=convert_bn_to_gn,
            is_frozen=is_frozen,
            frozen_keys_patterns=frozen_keys_patterns,
            unfrozen_keys_patterns=unfrozen_keys_patterns,
            pretrained_checkpoint=pretrained_checkpoint,
            **model_kwargs,
        )

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
            if (path / self.model_name).exists():
                cfg = Config.fromfile(path / self.model_name)
                for key, value in kwargs.items():
                    if hasattr(cfg.model, key):
                        setattr(cfg.model, key, value)
                return MODELS.build(cfg.model)

        raise FileNotFoundError(
            f"Model {self.model_name} not found in search paths {[str(x) for x in self._resolved_model_search_paths]}"
        )
