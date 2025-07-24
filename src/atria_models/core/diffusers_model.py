"""
Diffusers Model Builder Module

This module defines the `DiffusersModel` class and its subclasses, which provide functionality
for constructing models from the `diffusers` library. It supports tasks such as
image autoencoding, image generation, and diffusion-based models.

Classes:
    - DiffusersModel: A base model constructor for Diffusers models.
    - DiffusersAutoencoderModel: A specialized model constructor for autoencoder models in Diffusers.
    - DiffusersDiffusionModel: A specialized model constructor for diffusion models in Diffusers.

Dependencies:
    - torch: For PyTorch operations.
    - diffusers: For Diffusers model classes.
    - atria_core.logger: For logging utilities.
    - atria_core.utilities.common: For utility functions.
    - atria_core.utilities.imports: For resolving module paths.
    - atria_models.core.atria_model: For the base model class `AtriaModel`.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from typing import TYPE_CHECKING

from atria_core.logger import get_logger
from atria_core.utilities.imports import _resolve_module_from_path

from atria_models.core.atria_model import AtriaModel
from atria_models.registry import MODEL

if TYPE_CHECKING:
    from torch.nn import Module

logger = get_logger(__name__)


@MODEL.register("diffusers")
class DiffusersModel(AtriaModel):
    @property
    def model_name(self) -> str:
        """
        Get the name of the model.

        Returns:
            str: The name of the model.
        """
        return self.config.model_name

    def _resolve_model_class(self) -> type:
        """
        Resolve the model class from the Diffusers library.

        Returns:
            type: The resolved model class.

        Raises:
            Exception: If the model class cannot be resolved.
        """
        try:
            return _resolve_module_from_path(f"diffusers.{self.config.model_name}")
        except Exception as e:
            logger.exception(f"Error loading model class {self.config.model_name}: {e}")
            raise

    def _build(self, pretrained: bool = True, **kwargs) -> "Module":
        """
        Build the Diffusers model.

        This method resolves the model class, filters unused keyword arguments,
        and initializes the model.

        Returns:
            Module: The constructed Diffusers model.
        """
        model_class = self._resolve_model_class()
        model_config_name_or_path = kwargs.pop("model_config_name_or_path", None)
        if model_config_name_or_path:
            if pretrained:
                logger.info(
                    f"Initializing model [{model_class}] with the following parameters:"
                )
                logger.info(kwargs)

                return model_class.from_pretrained(
                    model_config_name_or_path, cache_dir=self._model_cache_dir, **kwargs
                )
            else:
                config = model_class.load_config(model_config_name_or_path)
                return model_class.from_config(config)
        else:
            return model_class(**kwargs)


@MODEL.register("diffusers/autoencoder")
class DiffusersAutoencoderModel(DiffusersModel):
    """
    A specialized model constructor for autoencoder models in Diffusers.

    This class is specifically designed for constructing autoencoder models
    such as `AutoencoderKL` from the Diffusers library.
    """

    def _build(self, **kwargs) -> "Module":
        """
        Build the autoencoder model.

        Ensures that the model name is valid for autoencoder models.

        Returns:
            Module: The constructed autoencoder model.
        """
        assert self.config.model_name in ["AutoencoderKL"]
        super()._build(subfolder="vae", **kwargs)


@MODEL.register("diffusers/diffusion_model")
class DiffusersDiffusionModel(DiffusersModel):
    """
    A specialized model constructor for diffusion models in Diffusers.

    This class is specifically designed for constructing diffusion models
    such as `UNet2DModel` and `UNet2DConditionModel` from the Diffusers library.
    """

    def _build(self, **kwargs) -> "Module":
        """
        Build the diffusion model.

        Ensures that the model name is valid for diffusion models.

        Returns:
            Module: The constructed diffusion model.
        """
        assert self.config.model_name in ["UNet2DModel", "UNet2DConditionModel"]
        super()._build(**kwargs)
