"""
Transformers Model Builder Module

This module defines the `TransformersModelBuilder` class, which provides functionality
for constructing models from the Hugging Face Transformers library. It supports tasks
such as sequence classification, token classification, image classification, and
question answering.

Classes:
    - TransformersModel: Base class for Hugging Face Transformers models.
    - SequenceClassificationTransformersModel: Model for sequence classification tasks.
    - TokenClassificationTransformersModel: Model for token classification tasks.
    - ImageClassificationTransformersModel: Model for image classification tasks.
    - QuestionAnsweringTransformersModel: Model for question answering tasks.

Dependencies:
    - hydra_zen: For configuration management.
    - transformers: For creating models from the Hugging Face Transformers library.
    - torch: For PyTorch operations.
    - atria_core.logger: For logging utilities.
    - atria_models.tasks: For defining model tasks.
    - atria_models.utilities.nn_modules: For neural network module utilities.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from typing import TYPE_CHECKING

from atria_core.constants import _DEFAULT_ATRIA_MODELS_CACHE_DIR
from atria_core.logger.logger import get_logger
from rich.pretty import pretty_repr

from atria_models.core.atria_model import AtriaModel
from atria_models.registry import MODEL

if TYPE_CHECKING:
    from torch.nn import Module

logger = get_logger(__name__)


@MODEL.register(
    "transformers",
    model_name_pattern="${.model_name}",  # take the model name from the relative '_here_' config
)
class TransformersModel(AtriaModel):
    """
    Base class for Hugging Face Transformers models.

    Args:
        model_name (str): Name of the Hugging Face model.
        model_cache_dir (Optional[str]): Directory for caching models.
        pretrained (bool): Whether to use pretrained weights.
        convert_bn_to_gn (bool): Whether to convert batch normalization to group normalization.
        is_frozen (bool): Whether to freeze the model parameters.
        frozen_keys_patterns (Optional[List[str]]): Patterns for keys to freeze.
        unfrozen_keys_patterns (Optional[List[str]]): Patterns for keys to unfreeze.
        **model_kwargs: Additional keyword arguments for the model.
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


@MODEL.register("transformers/sequence_classification")
class SequenceClassificationModel(TransformersModel):
    """
    Model for sequence classification tasks.

    This class builds a Hugging Face Transformers model for sequence classification.

    Returns:
        PreTrainedModel: Hugging Face model for sequence classification.
    """

    def _build(
        self, *, num_labels: int | None = None, pretrained: bool = True, **kwargs
    ) -> "Module":
        from transformers import AutoConfig, AutoModelForSequenceClassification

        if pretrained:
            if num_labels is not None:
                kwargs["num_labels"] = num_labels
            hf_config = AutoConfig.from_pretrained(
                self._model_name, cache_dir=self._model_cache_dir, **kwargs
            )

            logger.debug(
                f"Initializing the model with the following config:\n {pretty_repr(hf_config, expand_all=True)}"
            )
            return AutoModelForSequenceClassification.from_pretrained(
                self._model_name, config=hf_config, cache_dir=self._model_cache_dir
            )
        else:
            if num_labels is not None:
                kwargs["num_labels"] = num_labels
            hf_config = AutoConfig(
                self._model_name, cache_dir=self._model_cache_dir, **kwargs
            )
            return AutoModelForSequenceClassification(
                self._model_name, config=hf_config, cache_dir=self._model_cache_dir
            )


@MODEL.register("transformers/token_classification")
class TokenClassificationModel(TransformersModel):
    """
    Model for token classification tasks.

    This class builds a Hugging Face Transformers model for token classification.

    Returns:
        PreTrainedModel: Hugging Face model for token classification.
    """

    def _build(
        self, *, num_labels: int | None = None, pretrained: bool = True, **kwargs
    ) -> "Module":
        from transformers import AutoConfig, AutoModelForTokenClassification

        if pretrained:
            hf_config = AutoConfig.from_pretrained(
                self._model_name,
                cache_dir=self._model_cache_dir,
                num_labels=num_labels,
                **kwargs,
            )

            logger.debug(
                f"Initializing the model with the following config:\n {pretty_repr(hf_config, expand_all=True)}"
            )
            return AutoModelForTokenClassification.from_pretrained(
                self._model_name, config=hf_config, cache_dir=self._model_cache_dir
            )
        else:
            hf_config = AutoConfig(
                self._model_name,
                cache_dir=self._model_cache_dir,
                num_labels=num_labels,
                **kwargs,
            )
            return AutoModelForTokenClassification(
                self._model_name, config=hf_config, cache_dir=self._model_cache_dir
            )


@MODEL.register("transformers/image_classification")
class ImageClassificationModel(TransformersModel):
    """
    Model for image classification tasks.

    This class builds a Hugging Face Transformers model for image classification.

    Returns:
        PreTrainedModel: Hugging Face model for image classification.
    """

    def _build(
        self, *, num_labels: int | None = None, pretrained: bool = True, **kwargs
    ) -> "Module":
        from torch.nn import Linear
        from transformers import AutoConfig, AutoModelForImageClassification

        if pretrained:
            hf_config = AutoConfig.from_pretrained(
                self._model_name, cache_dir=self._model_cache_dir, **kwargs
            )

            logger.debug(
                f"Initializing the model with the following config:\n {pretty_repr(hf_config, expand_all=True)}"
            )
            model = AutoModelForImageClassification.from_pretrained(
                self._model_name, config=hf_config, cache_dir=self._model_cache_dir
            )
        else:
            hf_config = AutoConfig(
                self._model_name, cache_dir=self._model_cache_dir, **kwargs
            )
            model = AutoModelForImageClassification(
                self._model_name, config=hf_config, cache_dir=self._model_cache_dir
            )

        model.classifier = Linear(model.classifier.in_features, self._num_labels)
        logger.info(
            f"Initializing the classifier head for image classification with {model.classifier}"
        )
        return model


@MODEL.register("transformers/question_answering")
class QuestionAnsweringModel(TransformersModel):
    """
    Model for question answering tasks.

    This class builds a Hugging Face Transformers model for question answering.

    Returns:
        PreTrainedModel: Hugging Face model for question answering.
    """

    def _build(
        self, *, num_labels: int | None = None, pretrained: bool = True, **kwargs
    ) -> "Module":
        from transformers import AutoConfig, AutoModelForQuestionAnswering

        if pretrained:
            hf_config = AutoConfig.from_pretrained(
                self._model_name, cache_dir=self._model_cache_dir, **kwargs
            )

            logger.debug(
                f"Initializing the model with the following config:\n {pretty_repr(hf_config, expand_all=True)}"
            )
            return AutoModelForQuestionAnswering.from_pretrained(
                self._model_name, config=hf_config, cache_dir=self._model_cache_dir
            )
        else:
            hf_config = AutoConfig(
                self._model_name, cache_dir=self._model_cache_dir, **kwargs
            )
            return AutoModelForQuestionAnswering(
                self._model_name, config=hf_config, cache_dir=self._model_cache_dir
            )
