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

from atria_core.constants import _DEFAULT_ATRIA_MODELS_CACHE_DIR
from atria_core.logger import get_logger
from atria_core.utilities.common import _get_possible_args
from atria_core.utilities.imports import _resolve_module_from_path
from diffusers.models import AutoencoderKL
from torch.nn import Module

from atria_models.core.atria_model import AtriaModel
from atria_models.registry import MODEL

logger = get_logger(__name__)


@MODEL.register(
    "diffusers",
    model_name_pattern="${.model_name}",  # take the model name from the relative '_here_' config
)
class DiffusersModel(AtriaModel):
    """
    A base model constructor for Diffusers models.

    This class provides functionality for constructing models from the Diffusers
    library. It supports configurations such as pretraining, freezing layers, and
    filtering unused keyword arguments.

    Attributes:
        model_name (str): The name of the model to be constructed.
        num_labels (Optional[int]): The number of output labels for the model.
        model_cache_dir (Optional[str]): The directory where the model weights are stored.
        pretrained (bool): Whether to load pretrained weights.
        convert_bn_to_gn (bool): Whether to convert BatchNorm layers to GroupNorm layers.
        is_frozen (bool): Whether to freeze the model.
        frozen_keys_patterns (Optional[List[str]]): Patterns to freeze model layers.
        unfrozen_keys_patterns (Optional[List[str]]): Patterns to unfreeze model layers.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int | None = None,
        model_cache_dir: str | None = None,
        convert_bn_to_gn: bool = False,
        is_frozen: bool = False,
        frozen_keys_patterns: list[str] | None = None,
        unfrozen_keys_patterns: list[str] | None = None,
        pretrained_checkpoint: str | None = None,
        **model_kwargs,
    ):
        """
        Initialize the DiffusersModel instance.

        Args:
            model_name (str): The name of the model to be constructed.
            num_labels (Optional[int], optional): The number of output labels for the model. Defaults to None.
            model_cache_dir (Optional[str], optional): The directory where the model weights are stored. Defaults to None.
            pretrained (bool, optional): Whether to load pretrained weights. Defaults to True.
            convert_bn_to_gn (bool, optional): Whether to convert BatchNorm layers to GroupNorm layers. Defaults to False.
            is_frozen (bool, optional): Whether to freeze the model. Defaults to False.
            frozen_keys_patterns (Optional[List[str]], optional): Patterns to freeze model layers. Defaults to None.
            unfrozen_keys_patterns (Optional[List[str]], optional): Patterns to unfreeze model layers. Defaults to None.
            **model_kwargs: Additional keyword arguments for the model.
        """
        self._model_name = model_name
        self._num_labels = num_labels
        self._model_cache_dir = model_cache_dir or _DEFAULT_ATRIA_MODELS_CACHE_DIR

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
        Get the name of the model.

        Returns:
            str: The name of the model.
        """
        return self._model_name

    def _resolve_model_class(self):
        """
        Resolve the model class from the Diffusers library.

        Returns:
            type: The resolved model class.

        Raises:
            Exception: If the model class cannot be resolved.
        """
        try:
            return _resolve_module_from_path(f"diffusers.{self._model_name}")
        except Exception as e:
            logger.exception(f"Error loading model class {self._model_name}: {e}")
            raise

    def _filter_unused_kwargs(self, model_class):
        """
        Filter out unused keyword arguments for the model class.

        Args:
            model_class (type): The class of the model being constructed.
        """
        possible_args = _get_possible_args(model_class)
        leftover_args = set(self._model_kwargs.keys()) - set(possible_args)

        if leftover_args:
            logger.warning(
                f"The following parameters are not used in model initialization: {leftover_args}"
            )

        self._model_kwargs = {
            k: v for k, v in self._model_kwargs.items() if k in possible_args
        }

    def _initialize_model(self, model_class, pretrained: bool = True):
        """
        Initialize the model using the resolved model class.

        Args:
            model_class (type): The class of the model being constructed.

        Returns:
            Module: The initialized model.
        """
        model_config_name_or_path = self._model_kwargs.pop(
            "model_config_name_or_path", None
        )

        if model_config_name_or_path:
            if pretrained:
                additional_kwargs = {}

                if issubclass(model_class, AutoencoderKL):
                    additional_kwargs["subfolder"] = "vae"

                return model_class.from_pretrained(
                    model_config_name_or_path,
                    cache_dir=self._model_cache_dir,
                    **self._model_kwargs,
                    **additional_kwargs,
                )
            else:
                config = model_class.load_config(model_config_name_or_path)
                return model_class.from_config(config)
        else:
            return model_class(**self._model_kwargs)

    def _build(self, **kwargs) -> Module:
        """
        Build the Diffusers model.

        This method resolves the model class, filters unused keyword arguments,
        and initializes the model.

        Returns:
            Module: The constructed Diffusers model.
        """
        model_class = self._resolve_model_class()

        self._filter_unused_kwargs(model_class)

        logger.info(
            f"Initializing model [{model_class}] with the following parameters:"
        )
        logger.info(self._model_kwargs)

        return self._initialize_model(model_class)


@MODEL.register("diffusers/autoencoder")
class DiffusersAutoencoderModel(DiffusersModel):
    """
    A specialized model constructor for autoencoder models in Diffusers.

    This class is specifically designed for constructing autoencoder models
    such as `AutoencoderKL` from the Diffusers library.
    """

    def _build(self) -> Module:
        """
        Build the autoencoder model.

        Ensures that the model name is valid for autoencoder models.

        Returns:
            Module: The constructed autoencoder model.
        """
        assert self._model_name in ["AutoencoderKL"]
        super()._build()


@MODEL.register("diffusers/diffusion_model")
class DiffusersDiffusionModel(DiffusersModel):
    """
    A specialized model constructor for diffusion models in Diffusers.

    This class is specifically designed for constructing diffusion models
    such as `UNet2DModel` and `UNet2DConditionModel` from the Diffusers library.
    """

    def _build(self) -> Module:
        """
        Build the diffusion model.

        Ensures that the model name is valid for diffusion models.

        Returns:
            Module: The constructed diffusion model.
        """
        assert self._model_name in ["UNet2DModel", "UNet2DConditionModel"]
        super()._build()
