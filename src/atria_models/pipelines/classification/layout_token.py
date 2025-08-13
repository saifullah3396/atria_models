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

from atria_core.types import LayoutTokenClassificationModelOutput, TaskType

from atria_models.core.local_model import LocalModel
from atria_models.core.transformers_model import TokenClassificationModel
from atria_models.pipelines.atria_model_pipeline import AtriaModelPipelineConfig
from atria_models.pipelines.classification.token import TokenClassificationPipeline
from atria_models.pipelines.utilities import OverflowStrategy
from atria_models.registry import MODEL_PIPELINE

if TYPE_CHECKING:
    from typing import TYPE_CHECKING, Any

    from atria_transforms.data_types import TokenizedDocumentInstance


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
            "/data_transform@runtime_transforms.train": "document_instance_tokenizer/layout_token_classification"
        },
        {
            "/data_transform@runtime_transforms.evaluation": "document_instance_tokenizer/layout_token_classification"
        },
        {"/metric@metrics.layout_precision": "layout_precision"},
        {"/metric@metrics.layout_recall": "layout_recall"},
        {"/metric@metrics.layout_f1": "layout_f1"},
    ],
)
class LayoutTokenClassificationPipeline(TokenClassificationPipeline):
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
    __task_type__: TaskType = TaskType.layout_token_classification

    def _output_transform(
        self,
        batch: TokenizedDocumentInstance,
        model_output: LayoutTokenClassificationModelOutput,
    ) -> LayoutTokenClassificationModelOutput:
        """
        Transform the model output to the expected format for token classification.

        Args:
            batch (TokenizedDocumentInstance): The input batch of data.
            model_output (TokenClassificationModelOutput): The output from the model.

        Returns:
            TokenClassificationModelOutput: The transformed output with labels and logits.
        """
        # map the labels from the model output to the evaluation labels
        predicted_label_names, predicted_label_values = self._extract_predicted_labels(
            batch=batch, logits=model_output.logits
        )
        prediction_probs = self._extract_prediction_probs(
            logits=model_output.logits, mask=batch.prediction_indices_mask
        )

        batch.token_labels[~batch.prediction_indices_mask] = -100
        return LayoutTokenClassificationModelOutput(
            loss=model_output.loss,
            logits=model_output.logits,
            token_labels=batch.token_labels,
            token_bboxes=batch.token_bboxes,
            predicted_label_names=predicted_label_names,
            predicted_label_values=predicted_label_values,
            prediction_probs=prediction_probs,
            words=batch.words,
            word_bboxes=batch.word_bboxes.value,
            word_bboxes_mode=batch.word_bboxes.mode,
        )

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
