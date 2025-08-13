"""
Sequence Classification Pipeline Module

This module defines the `SequenceClassificationPipeline` class, which is a specific implementation
of the `ClassificationPipeline` for sequence classification tasks. It provides methods for training,
evaluation, and prediction, as well as utilities for handling tokenized sequence inputs.

Classes:
    - SequenceClassificationPipeline: A pipeline for sequence classification tasks.

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

from atria_core.types import ClassificationModelOutput, TaskType

from atria_models.core.local_model import LocalModel
from atria_models.core.transformers_model import SequenceClassificationModel
from atria_models.pipelines.atria_model_pipeline import AtriaModelPipelineConfig
from atria_models.pipelines.classification.base import ClassificationPipeline
from atria_models.pipelines.utilities import OverflowStrategy
from atria_models.registry import MODEL_PIPELINE

if TYPE_CHECKING:
    from atria_transforms.data_types import TokenizedDocumentInstance


class SequenceClassificationPipelineConfig(AtriaModelPipelineConfig):
    model: SequenceClassificationModel | LocalModel
    use_bbox: bool = True
    use_image: bool = True
    training_overflow_strategy: OverflowStrategy = OverflowStrategy.select_first
    evaluation_overflow_strategy: OverflowStrategy = OverflowStrategy.select_first
    input_stride: int = 0


@MODEL_PIPELINE.register(
    "sequence_classification",
    defaults=[
        "_self_",
        {"/model@model": "transformers/sequence_classification"},
        {
            "/data_transform@runtime_transforms.train": "document_instance_tokenizer/sequence_classification"
        },
        {
            "/data_transform@runtime_transforms.evaluation": "document_instance_tokenizer/sequence_classification"
        },
        {"/metric@metrics.accuracy": "accuracy"},
        {"/metric@metrics.precision": "precision"},
        {"/metric@metrics.recall": "recall"},
        {"/metric@metrics.f1_score": "f1_score"},
        {"/metric@metrics.confusion_matrix": "confusion_matrix"},
    ],
)
class SequenceClassificationPipeline(ClassificationPipeline):
    """
    A pipeline for sequence classification tasks.

    This class extends the `ClassificationPipeline` to provide functionality specific to
    sequence classification. It supports training, evaluation, and prediction steps, and
    handles tokenized sequence inputs with optional bounding box and image data.

    Attributes:
        model (Union[AtriaModel, Dict[str, AtriaModel]]): The model or dictionary of models.
        checkpoint_configs (Optional[List[CheckpointConfig]]): Configuration for model checkpoints.
        use_bbox (bool): Flag indicating whether bounding box data is used.
        use_image (bool): Flag indicating whether image data is used.
    """

    __config_cls__ = SequenceClassificationPipelineConfig
    __task_type__: TaskType = TaskType.sequence_classification

    def can_be_evaluated(self, batch: TokenizedDocumentInstance) -> bool:
        return batch.label is not None

    def training_step(
        self, batch: TokenizedDocumentInstance, **kwargs
    ) -> ClassificationModelOutput:
        """
        Performs a single training step.

        Args:
            batch (TokenizedDocumentInstance): The input batch of data.
            **kwargs: Additional arguments.

        Returns:
            ClassificationModelOutput: The output of the training step, including loss and logits.
        """
        from atria_core.types import ClassificationModelOutput

        from atria_models.utilities.nn_modules import _get_logits_from_output

        if self.config.training_overflow_strategy == OverflowStrategy.select_all:
            batch.select_all_overflow_samples()
        elif self.config.training_overflow_strategy == OverflowStrategy.select_random:
            batch.select_random_overflow_samples()
        elif self.config.training_overflow_strategy == OverflowStrategy.select_first:
            batch.select_first_overflow_samples()

        logits = _get_logits_from_output(self._model_forward(batch))
        loss = self._loss_fn_train(logits, batch.label.value)
        predicted_labels = logits.argmax(dim=-1)
        return ClassificationModelOutput(
            loss=loss,
            logits=logits,
            prediction_probs=logits.softmax(dim=-1),
            gt_label_value=batch.label.value,
            gt_label_name=batch.label.name,
            predicted_label_value=predicted_labels,
            predicted_label_name=[
                self._dataset_metadata.dataset_labels.classification[i]
                for i in predicted_labels.tolist()
            ],
        )

    def evaluation_step(
        self, batch: TokenizedDocumentInstance, **kwargs
    ) -> ClassificationModelOutput:
        """
        Performs a single evaluation step.

        Args:
            batch (TokenizedDocumentInstance): The input batch of data.
            **kwargs: Additional arguments.

        Returns:
            ClassificationModelOutput: The output of the evaluation step, including loss and logits.
        """
        from atria_core.types import ClassificationModelOutput

        from atria_models.utilities.nn_modules import _get_logits_from_output

        if self.config.evaluation_overflow_strategy == OverflowStrategy.select_all:
            batch.select_all_overflow_samples()
        elif self.config.evaluation_overflow_strategy == OverflowStrategy.select_random:
            batch.select_random_overflow_samples()
        elif self.config.evaluation_overflow_strategy == OverflowStrategy.select_first:
            batch.select_first_overflow_samples()

        logits = _get_logits_from_output(self._model_forward(batch))
        loss = self._loss_fn_eval(logits, batch.label.value)
        predicted_labels = logits.argmax(dim=-1)
        return ClassificationModelOutput(
            loss=loss,
            logits=logits,
            prediction_probs=logits.softmax(dim=-1),
            gt_label_value=batch.label.value,
            gt_label_name=batch.label.name,
            predicted_label_value=predicted_labels,
            predicted_label_name=[
                self._dataset_metadata.dataset_labels.classification[i]
                for i in predicted_labels.tolist()
            ],
        )

    def predict_step(
        self, batch: TokenizedDocumentInstance, **kwargs
    ) -> ClassificationModelOutput:
        """
        Performs a single prediction step.

        Args:
            batch (TokenizedDocumentInstance): The input batch of data.
            **kwargs: Additional arguments.

        Returns:
            ClassificationModelOutput: The output of the prediction step, including logits and predictions.
        """
        from atria_core.types import ClassificationModelOutput

        from atria_models.utilities.nn_modules import _get_logits_from_output

        if self.config.evaluation_overflow_strategy == OverflowStrategy.select_all:
            batch.select_all_overflow_samples()
        elif self.config.evaluation_overflow_strategy == OverflowStrategy.select_random:
            batch.select_random_overflow_samples()
        elif self.config.evaluation_overflow_strategy == OverflowStrategy.select_first:
            batch.select_first_overflow_samples()

        logits = _get_logits_from_output(self._model_forward(batch))
        predicted_labels = logits.argmax(dim=-1)
        return ClassificationModelOutput(
            logits=logits,
            prediction_probs=logits.softmax(dim=-1),
            predicted_label_value=predicted_labels,
            predicted_label_name=[
                self._dataset_metadata.dataset_labels.classification[i]
                for i in predicted_labels.tolist()
            ],
        )

    def _model_forward(self, batch: TokenizedDocumentInstance) -> Any:
        """
        Perform a forward pass through the model.

        Args:
            batch (TokenizedObjectInstance): A batch of tokenized object instances.

        Returns:
            Any: The output of the model forward pass.
        """
        if batch.token_bboxes.min() < 1.0:
            batch.token_bboxes = (batch.token_bboxes * 1000.0).long()
        inputs = {
            "input_ids": batch.token_ids,
            "token_type_ids": batch.token_type_ids,
            "attention_mask": batch.attention_mask,
            "bbox": batch.token_bboxes if self.config.use_bbox else None,
            "pixel_values": batch.image.content if self.config.use_image else None,
            "labels": batch.label.value,
        }
        return self._model(**inputs)
