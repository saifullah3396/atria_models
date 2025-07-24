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

from typing import Any

from atria_core.transforms import DataTransformsDict
from atria_core.types import TaskType
from atria_transforms.data_types import TokenizedDocumentInstance

from atria_models.core.local_model import LocalModel
from atria_models.core.transformers_model import SequenceClassificationModel
from atria_models.data_types.outputs import ClassificationModelOutput
from atria_models.pipelines.atria_model_pipeline import MetricInitializer
from atria_models.pipelines.classification.base import ClassificationPipeline
from atria_models.pipelines.utilities import OverflowStrategy
from atria_models.registry import MODEL_PIPELINE
from atria_models.utilities.checkpoints import CheckpointConfig


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
    ],
    metrics=[
        MetricInitializer(name="accuracy"),
        MetricInitializer(name="precision"),
        MetricInitializer(name="recall"),
        MetricInitializer(name="f1_score"),
        MetricInitializer(name="confusion_matrix"),
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

    _TASK_TYPE: TaskType = TaskType.sequence_classification

    def __init__(
        self,
        model: SequenceClassificationModel | LocalModel,
        checkpoint_configs: list[CheckpointConfig] | None = None,
        metrics: list[MetricInitializer] | None = None,
        runtime_transforms: DataTransformsDict = DataTransformsDict(),
        use_bbox: bool = True,
        use_image: bool = True,
        training_overflow_strategy: OverflowStrategy
        | None = OverflowStrategy.select_first,
        evaluation_overflow_strategy: OverflowStrategy
        | None = OverflowStrategy.select_first,
    ):
        """
        Initialize the SequenceClassificationPipeline.

        Args:
            model (Union[AtriaModel, Dict[str, AtriaModel]]): The model or dictionary of models.
            checkpoint_configs (Optional[List[CheckpointConfig]]): Configuration for model checkpoints.
            use_bbox (bool): Flag indicating whether bounding box data is used.
            use_image (bool): Flag indicating whether image data is used.
        """
        self._use_bbox = use_bbox
        self._use_image = use_image
        self._training_overflow_strategy = training_overflow_strategy
        self._evaluation_overflow_strategy = evaluation_overflow_strategy

        super().__init__(
            model=model,
            checkpoint_configs=checkpoint_configs,
            metric_configs=metrics,
            runtime_transforms=runtime_transforms,
        )

    def training_step(
        self, batch: TokenizedDocumentInstance, **kwargs
    ) -> "ClassificationModelOutput":
        """
        Performs a single training step.

        Args:
            batch (TokenizedDocumentInstance): The input batch of data.
            **kwargs: Additional arguments.

        Returns:
            ClassificationModelOutput: The output of the training step, including loss and logits.
        """
        from atria_models.data_types.outputs import ClassificationModelOutput

        if self._training_overflow_strategy == OverflowStrategy.select_all:
            batch.select_all_overflow_samples()
        elif self._training_overflow_strategy == OverflowStrategy.select_random:
            batch.select_random_overflow_samples()
        elif self._training_overflow_strategy == OverflowStrategy.select_first:
            batch.select_first_overflow_samples()

        logits = _get_logits_from_output(self._model_forward(batch))
        loss = self._loss_fn_train(logits, batch.label)
        return ClassificationModelOutput(loss=loss, logits=logits, label=batch.label)

    def evaluation_step(
        self, batch: TokenizedDocumentInstance, **kwargs
    ) -> "ClassificationModelOutput":
        """
        Performs a single evaluation step.

        Args:
            batch (TokenizedDocumentInstance): The input batch of data.
            **kwargs: Additional arguments.

        Returns:
            ClassificationModelOutput: The output of the evaluation step, including loss and logits.
        """
        from atria_models.data_types.outputs import ClassificationModelOutput

        if self._evaluation_overflow_strategy == OverflowStrategy.select_all:
            batch.select_all_overflow_samples()
        elif self._evaluation_overflow_strategy == OverflowStrategy.select_random:
            batch.select_random_overflow_samples()
        elif self._evaluation_overflow_strategy == OverflowStrategy.select_first:
            batch.select_first_overflow_samples()

        logits = _get_logits_from_output(self._model_forward(batch))
        loss = self._loss_fn_eval(logits, batch.label)
        return ClassificationModelOutput(loss=loss, logits=logits, label=batch.label)

    def predict_step(
        self, batch: TokenizedDocumentInstance, **kwargs
    ) -> "ClassificationModelOutput":
        """
        Performs a single prediction step.

        Args:
            batch (TokenizedDocumentInstance): The input batch of data.
            **kwargs: Additional arguments.

        Returns:
            ClassificationModelOutput: The output of the prediction step, including logits and predictions.
        """
        from atria_models.data_types.outputs import ClassificationModelOutput

        if self._evaluation_overflow_strategy == OverflowStrategy.select_all:
            batch.select_all_overflow_samples()
        elif self._evaluation_overflow_strategy == OverflowStrategy.select_random:
            batch.select_random_overflow_samples()
        elif self._evaluation_overflow_strategy == OverflowStrategy.select_first:
            batch.select_first_overflow_samples()

        logits = _get_logits_from_output(self._model_forward(batch))
        return ClassificationModelOutput(
            logits=logits, prediction=logits.argmax(dim=-1)
        )

    def _model_forward(self, batch: TokenizedDocumentInstance) -> Any:
        """
        Perform a forward pass through the model.

        Args:
            batch (TokenizedObjectInstance): A batch of tokenized object instances.

        Returns:
            Any: The output of the model forward pass.
        """
        batch.token_bboxes = (batch.token_bboxes * 1000.0).long()
        inputs = {
            "input_ids": batch.token_ids,
            "token_type_ids": batch.token_type_ids,
            "attention_mask": batch.attention_mask,
            "bbox": batch.token_bboxes if self._use_bbox else None,
            "pixel_values": batch.image.content if self._use_image else None,
            "labels": batch.label,
        }
        return self._model(**inputs)
