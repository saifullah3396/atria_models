"""
Classification Pipeline Module

This module defines the `ClassificationPipeline` class, which serves as a base class
for implementing classification tasks using PyTorch models. It provides methods for
training, evaluation, and prediction, as well as utilities for preparing model
configuration and performing forward passes.

Classes:
    - ClassificationPipeline: Base class for classification tasks.

Dependencies:
    - torch: For PyTorch operations.
    - atria_core.logger: For logging utilities.
    - atria_models.utilities: For neural network utilities.
    - atria_models.data_types.outputs: For model output structures.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from atria_core.logger import get_logger
from atria_core.types import DocumentInstance, ImageInstance
from atria_models.pipelines.atria_model_pipeline import AtriaModelPipeline

if TYPE_CHECKING:
    from atria_core.types import ClassificationModelOutput
    from atria_models.core.atria_model import AtriaModel

logger = get_logger(__name__)

SupportedBatchDataTypes = DocumentInstance | ImageInstance


class ClassificationPipeline(AtriaModelPipeline):
    """
    Base class for classification tasks.

    This class provides methods for training, evaluation, and prediction, as well as
    utilities for preparing model configuration and performing forward passes.

    Attributes:
        _loss_fn_train: Loss function used during training.
        _loss_fn_eval: Loss function used during evaluation.
    """

    def can_be_evaluated(self, batch: SupportedBatchDataTypes) -> bool:
        return batch.gt.classification is not None

    def training_step(
        self, batch: SupportedBatchDataTypes, **kwargs
    ) -> ClassificationModelOutput:
        """
        Performs a single training step.

        Args:
            batch (SupportedBatchDataTypes): The input batch of data.
            **kwargs: Additional arguments.

        Returns:
            ClassificationModelOutput: The output of the training step, including loss and logits.
        """

        from atria_core.types import ClassificationModelOutput
        from atria_models.utilities.nn_modules import _get_logits_from_output

        logits = _get_logits_from_output(self._model_forward(batch))
        loss = self._loss_fn_train(logits, batch.gt.classification.label.value)
        return ClassificationModelOutput(
            loss=loss, logits=logits, label=batch.gt.classification.label
        )

    def evaluation_step(
        self, batch: SupportedBatchDataTypes, **kwargs
    ) -> ClassificationModelOutput:
        """
        Performs a single evaluation step.

        Args:
            batch (SupportedBatchDataTypes): The input batch of data.
            **kwargs: Additional arguments.

        Returns:
            ClassificationModelOutput: The output of the evaluation step, including loss and logits.
        """

        from atria_core.types import ClassificationModelOutput
        from atria_models.utilities.nn_modules import _get_logits_from_output

        logits = _get_logits_from_output(self._model_forward(batch))
        loss = self._loss_fn_eval(logits, batch.gt.classification.label.value)
        predicted_labels = logits.argmax(dim=-1)
        return ClassificationModelOutput(
            loss=loss,
            logits=logits,
            gt_label=batch.gt.classification.label.value,
            gt_label_name=batch.gt.classification.label.name,
            predicted_label=predicted_labels,
            predicted_label_name=[
                self._dataset_metadata.dataset_labels.classification[i]
                for i in predicted_labels.tolist()
            ],
        )

    def predict_step(
        self, batch: SupportedBatchDataTypes, **kwargs
    ) -> ClassificationModelOutput:
        """
        Performs a single prediction step.

        Args:
            batch (SupportedBatchDataTypes): The input batch of data.
            **kwargs: Additional arguments.

        Returns:
            ClassificationModelOutput: The output of the prediction step, including logits and predictions.
        """

        from atria_core.types import ClassificationModelOutput
        from atria_models.utilities.nn_modules import _get_logits_from_output

        logits = _get_logits_from_output(self._model_forward(batch))
        predicted_labels = logits.argmax(dim=-1)
        return ClassificationModelOutput(
            logits=logits,
            gt_label=batch.gt.classification.label.value,
            gt_label_name=batch.gt.classification.label.name,
            predicted_label=predicted_labels,
            predicted_label_name=[
                self._dataset_metadata.dataset_labels.classification[i]
                for i in predicted_labels.tolist()
            ],
        )

    def _build_model(self) -> AtriaModel:
        import torch

        self._loss_fn_train = torch.nn.CrossEntropyLoss()
        self._loss_fn_eval = torch.nn.CrossEntropyLoss()
        return super()._build_model()

    def _prepare_build_kwargs(self) -> dict[str, dict[str, Any]] | dict[str, Any]:
        """
        Prepares keyword arguments for building the model.

        Returns:
            Dict[str, Dict[str, Any]] | Dict[str, Any]: The prepared keyword arguments.
        """

        if self._dataset_metadata is None:
            return {}
        assert self._dataset_metadata.dataset_labels.classification is not None, (
            f"`instance_classification` dataset labels must be provided for {self.__class__.__name__}."
            f"Dataset labels found in metadata: {self._dataset_metadata.dataset_labels}"
        )
        labels = self._dataset_metadata.dataset_labels.classification
        if isinstance(self._model, dict):
            return {key: {"num_labels": len(labels)} for key in self._model}
        else:
            return {"num_labels": len(labels)}

    @abstractmethod
    def _model_forward(self, batch: SupportedBatchDataTypes) -> Any:
        """
        Forward pass of the model.

        Args:
            batch (SupportedBatchDataTypes): The input batch of data.

        Returns:
            Any: The output of the model.
        """

        raise NotImplementedError(
            "The _model_forward method must be implemented in subclasses."
        )
