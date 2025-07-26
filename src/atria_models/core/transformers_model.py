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

from atria_core.logger.logger import get_logger
from rich.pretty import pretty_repr

from atria_models.core.atria_model import AtriaModel, AtriaModelConfig
from atria_models.registry import MODEL

if TYPE_CHECKING:
    from torch.nn import Module

logger = get_logger(__name__)


class TransformersModelConfig(AtriaModelConfig):
    tf_name: str = "???"  # Placeholder for the model name


class TransformersModel(AtriaModel):
    __config_cls__ = TransformersModelConfig


@MODEL.register("transformers/sequence_classification")
class SequenceClassificationModel(TransformersModel):
    """
    Model for sequence classification tasks.

    This class builds a Hugging Face Transformers model for sequence classification.

    Returns:
        PreTrainedModel: Hugging Face model for sequence classification.
    """

    __config_cls__ = TransformersModelConfig

    def _build(
        self, *, num_labels: int | None = None, pretrained: bool = True, **kwargs
    ) -> "Module":
        from transformers import AutoConfig, AutoModelForSequenceClassification

        if pretrained:
            if num_labels is not None:
                kwargs["num_labels"] = num_labels
            hf_config = AutoConfig.from_pretrained(
                self.config.tf_name, cache_dir=self.config.model_cache_dir, **kwargs
            )

            logger.debug(
                f"Initializing the model with the following config:\n {pretty_repr(hf_config, expand_all=True)}"
            )
            return AutoModelForSequenceClassification.from_pretrained(
                self.config.tf_name,
                config=hf_config,
                cache_dir=self.config.model_cache_dir,
            )
        else:
            if num_labels is not None:
                kwargs["num_labels"] = num_labels
            hf_config = AutoConfig(
                self.config.tf_name, cache_dir=self.config.model_cache_dir, **kwargs
            )
            return AutoModelForSequenceClassification(
                self.config.tf_name,
                config=hf_config,
                cache_dir=self.config.model_cache_dir,
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
                self.config.tf_name,
                cache_dir=self.config.model_cache_dir,
                num_labels=num_labels,
                **kwargs,
            )

            logger.debug(
                f"Initializing the model with the following config:\n {pretty_repr(hf_config, expand_all=True)}"
            )
            return AutoModelForTokenClassification.from_pretrained(
                self.config.tf_name,
                config=hf_config,
                cache_dir=self.config.model_cache_dir,
            )
        else:
            hf_config = AutoConfig(
                self.config.tf_name,
                cache_dir=self.config.model_cache_dir,
                num_labels=num_labels,
                **kwargs,
            )
            return AutoModelForTokenClassification(
                self.config.tf_name,
                config=hf_config,
                cache_dir=self.config.model_cache_dir,
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
                self.config.tf_name, cache_dir=self.config.model_cache_dir, **kwargs
            )

            logger.debug(
                f"Initializing the model with the following config:\n {pretty_repr(hf_config, expand_all=True)}"
            )
            model = AutoModelForImageClassification.from_pretrained(
                self.config.tf_name,
                config=hf_config,
                cache_dir=self.config.model_cache_dir,
            )
        else:
            hf_config = AutoConfig(
                self.config.tf_name, cache_dir=self.config.model_cache_dir, **kwargs
            )
            model = AutoModelForImageClassification(
                self.config.tf_name,
                config=hf_config,
                cache_dir=self.config.model_cache_dir,
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
                self.config.tf_name, cache_dir=self.config.model_cache_dir, **kwargs
            )

            logger.debug(
                f"Initializing the model with the following config:\n {pretty_repr(hf_config, expand_all=True)}"
            )
            return AutoModelForQuestionAnswering.from_pretrained(
                self.config.tf_name,
                config=hf_config,
                cache_dir=self.config.model_cache_dir,
            )
        else:
            hf_config = AutoConfig(
                self.config.tf_name, cache_dir=self.config.model_cache_dir, **kwargs
            )
            return AutoModelForQuestionAnswering(
                self.config.tf_name,
                config=hf_config,
                cache_dir=self.config.model_cache_dir,
            )
