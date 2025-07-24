"""
Layout Token Classification Pipeline Module

This module defines the `LayoutTokenClassificationPipeline` class, which is a specific implementation
of the `ClassificationPipeline` for layout token classification tasks. It provides methods for training,
evaluation, and prediction, as well as utilities for handling tokenized inputs with layout information.

Classes:
    - LayoutTokenClassificationPipeline: A pipeline for layout token classification tasks.

Dependencies:
    - torch: For PyTorch operations.
    - ignite: For distributed training and logging utilities.
    - atria_models.utilities: For checkpoint management and neural network utilities.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from atria_core.types import TaskType
from atria_registry.registry_config import RegistryConfig

from atria_models.core.local_model import LocalModel
from atria_models.core.transformers_model import TokenClassificationModel
from atria_models.pipelines.atria_model_pipeline import AtriaModelPipelineConfig
from atria_models.pipelines.classification.base import ClassificationPipeline
from atria_models.pipelines.utilities import OverflowStrategy
from atria_models.registry import MODEL_PIPELINE

if TYPE_CHECKING:
    from typing import TYPE_CHECKING, Any

    import torch
    from atria_transforms.data_types import TokenizedDocumentInstance

    from atria_models.data_types.outputs import LayoutTokenClassificationModelOutput


class SequenceClassificationPipelineConfig(AtriaModelPipelineConfig):
    model: TokenClassificationModel | LocalModel
    use_bbox: bool = True
    use_image: bool = True
    training_overflow_strategy: OverflowStrategy = OverflowStrategy.select_random
    evaluation_overflow_strategy: OverflowStrategy = OverflowStrategy.select_all
    input_stride: int = 0


@MODEL_PIPELINE.register(
    "layout_token_classification",
    defaults=[
        "_self_",
        {"/model@model": "transformers/token_classification"},
        {
            "/data_transform@runtime_transforms.train": "document_instance_tokenizer/sequence_classification"
        },
        {
            "/data_transform@runtime_transforms.evaluation": "document_instance_tokenizer/sequence_classification"
        },
    ],
    metric_configs=[
        RegistryConfig(name="layout_precision"),
        RegistryConfig(name="layout_recall"),
        RegistryConfig(name="layout_f1"),
    ],
)
class LayoutTokenClassificationPipeline(ClassificationPipeline):
    """
    A pipeline for layout token classification tasks.

    This class extends the `ClassificationPipeline` to handle tokenized inputs with layout information,
    such as bounding boxes and pixel values. It provides methods for training, evaluation, and prediction,
    as well as utilities for preparing model inputs and handling strided tokenization.

    Attributes:
        - model_factory: A factory function or dictionary of factory functions for creating the model(s).
        - dataset_metadata: Metadata for the dataset, including labels and other information.
        - checkpoint_configs: Configuration for model checkpoints.
        - tb_logger: Tensorboard logger for tracking training metrics.
        - use_bbox: Whether to use bounding box information in the model inputs.
        - use_image: Whether to use image pixel values in the model inputs.
        - input_stride: The stride value for handling overlapping tokens in strided tokenization.
    """

    __config_cls__ = SequenceClassificationPipelineConfig
    __task_type__: TaskType = TaskType.layout_entity_recognition

    def _remove_predictions_for_strided_input(self, batch: TokenizedDocumentInstance):
        """
        Fix the input stride for the batch.

        Args:
            batch (TokenizedDocumentInstance): The input batch of data.
        """
        # here we check if the input is strided or not. With strided input tokenization, the first "number of stride"
        # tokens are to be ignored for evaluation as they will be repeated tokens from the previous part of the document
        # first we check if there are overflowing samples in the batch and if so for these tokens first N stride tokens
        # are to be ignored for evaluation
        if self.config.input_stride > 0:
            for sample_idx, sample_word_ids in enumerate(batch.word_ids):
                # if the minimum word id is greater than 0, then we have an overflowing sample
                # this means this is a continuation of the previous sample and the first N tokens
                if sample_word_ids[sample_word_ids != -100].min() > 0:
                    batch.prediction_indices_mask[sample_idx][
                        : self.config.input_stride
                    ] = False

    def _extract_target_labels(self, batch: TokenizedDocumentInstance):
        target_labels = []
        for target, mask in zip(
            batch.token_labels, batch.prediction_indices_mask, strict=True
        ):
            target_labels.append(
                [self._dataset_metadata.dataset_labels.ser[i] for i in target[mask]]
            )
        return target_labels

    def _extract_predicted_labels(
        self, batch: TokenizedDocumentInstance, logits: torch.Tensor
    ):
        predicted_labels = []
        for predicted, mask in zip(
            logits.argmax(-1), batch.prediction_indices_mask, strict=True
        ):
            predicted_labels.append(
                [self._dataset_metadata.dataset_labels.ser[i] for i in predicted[mask]]
            )
        return predicted_labels

    def training_step(
        self, batch: TokenizedDocumentInstance, **kwargs
    ) -> LayoutTokenClassificationModelOutput:
        """
        Performs a single training step.

        Args:
            batch (TokenizedDocumentInstance): The input batch of data.
            **kwargs: Additional arguments.

        Returns:
            ClassificationModelOutput: The output of the training step, including loss and logits.
        """

        if self.config.training_overflow_strategy == OverflowStrategy.select_all:
            batch.select_all_overflow_samples()
        elif self.config.training_overflow_strategy == OverflowStrategy.select_random:
            batch.select_random_overflow_samples()
        elif self.config.training_overflow_strategy == OverflowStrategy.select_first:
            batch.select_first_overflow_samples()

        output = self._model_forward(batch)
        logits = output.logits
        loss = output.loss

        # map the labels from the model output to the evaluation labels
        predicted_labels = self._extract_predicted_labels(batch=batch, logits=logits)

        return LayoutTokenClassificationModelOutput(
            loss=loss,
            logits=logits,
            token_labels=batch.token_labels,
            token_bboxes=batch.token_bboxes,
            predicted_labels=predicted_labels,
        )

    def evaluation_step(
        self, batch: TokenizedDocumentInstance, **kwargs
    ) -> LayoutTokenClassificationModelOutput:
        """
        Performs a single evaluation step.

        Args:
            batch (TokenizedDocumentInstance): The input batch of data.
            **kwargs: Additional arguments.

        Returns:
            ClassificationModelOutput: The output of the evaluation step, including loss and logits.
        """
        if self.config.evaluation_overflow_strategy == OverflowStrategy.select_all:
            batch.select_all_overflow_samples()
        elif self.config.evaluation_overflow_strategy == OverflowStrategy.select_random:
            batch.select_random_overflow_samples()
        elif self.config.evaluation_overflow_strategy == OverflowStrategy.select_first:
            batch.select_first_overflow_samples()
        output = self._model_forward(batch)
        logits = output.logits
        loss = output.loss

        # set prediction_indices to false for strided input tokens
        self._remove_predictions_for_strided_input(batch)

        # map the labels from the model output to the evaluation labels
        batch.token_labels[~batch.prediction_indices_mask] = -100
        predicted_labels = self._extract_predicted_labels(batch=batch, logits=logits)

        return LayoutTokenClassificationModelOutput(
            loss=loss,
            logits=logits,
            token_labels=batch.token_labels,
            token_bboxes=batch.token_bboxes,
            predicted_labels=predicted_labels,
        )

    def predict_step(
        self, batch: TokenizedDocumentInstance, **kwargs
    ) -> LayoutTokenClassificationModelOutput:
        """
        Performs a single prediction step.

        Args:
            batch (TokenizedDocumentInstance): The input batch of data.
            **kwargs: Additional arguments.

        Returns:
            ClassificationModelOutput: The output of the prediction step, including logits and predictions.
        """

        if self.config.evaluation_overflow_strategy == OverflowStrategy.select_all:
            batch.select_all_overflow_samples()
        elif self.config.evaluation_overflow_strategy == OverflowStrategy.select_random:
            batch.select_random_overflow_samples()
        elif self.config.evaluation_overflow_strategy == OverflowStrategy.select_first:
            batch.select_first_overflow_samples()

        # map the labels from the model output to the evaluation labels
        output = self._model_forward(batch)
        logits = output.logits
        loss = output.loss
        predicted_labels = self._extract_predicted_labels(batch=batch, logits=logits)
        return LayoutTokenClassificationModelOutput(
            loss=loss,
            logits=logits,
            token_labels=batch.token_labels,
            token_bboxes=batch.token_bboxes,
            predicted_labels=predicted_labels,
        )

    def _prepare_build_kwargs(self) -> dict[str, dict[str, Any]] | dict[str, Any]:
        """
        Prepares keyword arguments for building the model.

        Returns:
            Dict[str, Dict[str, Any]] | Dict[str, Any]: The prepared keyword arguments.
        """
        if self._dataset_metadata is None:
            return {}
        assert self._dataset_metadata.dataset_labels.ser is not None, (
            f"`semantic_entity_recognition` dataset labels must be provided for {self.__class__.__name__}."
            f"Dataset labels found in metadata: {self._dataset_metadata.dataset_labels}"
        )
        labels = self._dataset_metadata.dataset_labels.ser
        if isinstance(self._model, dict):
            return {key: {"num_labels": len(labels)} for key in self._model}
        else:
            return {"num_labels": len(labels)}

    def _model_forward(self, batch: TokenizedDocumentInstance) -> Any:
        """
        Perform a forward pass through the model.

        Args:
            batch (TokenizedObjectInstance): A batch of tokenized object instances.

        Returns:
            Any: The output of the model forward pass.
        """
        if batch.token_bboxes.max() > 0.0 and batch.token_bboxes.max() < 1.0:
            batch.token_bboxes = (batch.token_bboxes * 1000.0).long()
        else:
            batch.token_bboxes = batch.token_bboxes.long()
        inputs = {
            "input_ids": batch.token_ids,
            "token_type_ids": batch.token_type_ids,
            "attention_mask": batch.attention_mask,
            "bbox": batch.token_bboxes if self.config.use_bbox else None,
            "pixel_values": batch.image.content if self.config.use_image else None,
            "labels": batch.token_labels,
        }
        return self._model(**inputs)
