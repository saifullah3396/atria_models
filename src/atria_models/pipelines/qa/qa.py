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
    - atria.data: For dataset metadata and data structures.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from typing import Any

import torch
from atria_core.transforms import DataTransformsDict
from atria_core.types import TaskType
from atria_transforms.data_types import TokenizedDocumentInstance

from atria_models.core.transformers_model import QuestionAnsweringModel
from atria_models.data_types.outputs import SequenceQAModelOutput
from atria_models.pipelines.atria_model_pipeline import (
    AtriaModelPipeline,
    MetricInitializer,
)
from atria_models.pipelines.utilities import OverflowStrategy
from atria_models.registry import MODEL_PIPELINE
from atria_models.utilities.checkpoints import CheckpointConfig


@MODEL_PIPELINE.register(
    "visual_question_answering",
    defaults=[
        "_self_",
        {"/model@model": "transformers/question_answering"},
        {
            "/data_transform@runtime_transforms.train": "document_instance_tokenizer/visual_question_answering"
        },
        {
            "/data_transform@runtime_transforms.evaluation": "document_instance_tokenizer/visual_question_answering"
        },
    ],
    metrics=[MetricInitializer(name="sequence_anls")],
)
class QuestionAnsweringPipeline(AtriaModelPipeline):
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

    _TASK_TYPE: TaskType = TaskType.visual_question_answering

    def __init__(
        self,
        model: QuestionAnsweringModel,
        checkpoint_configs: list[CheckpointConfig] | None = None,
        metrics: list[MetricInitializer] | None = None,
        runtime_transforms: DataTransformsDict = DataTransformsDict(),
        use_bbox: bool = True,
        use_image: bool = True,
        training_overflow_strategy: OverflowStrategy
        | None = OverflowStrategy.select_random,
        evaluation_overflow_strategy: OverflowStrategy
        | None = OverflowStrategy.select_all,
        input_stride: int = 0,
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
        self._input_stride = input_stride

        super().__init__(
            model=model,
            checkpoint_configs=checkpoint_configs,
            metric_configs=metrics,
            runtime_transforms=runtime_transforms,
        )

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
        if self._input_stride > 0:
            for sample_idx, sample_word_ids in enumerate(batch.word_ids):
                # if the minimum word id is greater than 0, then we have an overflowing sample
                # this means this is a continuation of the previous sample and the first N tokens
                if sample_word_ids[sample_word_ids != -100].min() > 0:
                    batch.prediction_indices_mask[sample_idx][: self._input_stride] = (
                        False
                    )

    def _extract_target_labels(self, batch: TokenizedDocumentInstance):
        target_labels = []
        for target, mask in zip(
            batch.token_labels, batch.prediction_indices_mask, strict=True
        ):
            target_labels.append(
                [self._dataset_metadata.dataset_labels.ser[i] for i in target[mask]]
            )
        return target_labels

    def _extract_predicted_answers(
        self,
        batch: TokenizedDocumentInstance,
        start_logits: "torch.Tensor",
        end_logits: "torch.Tensor",
    ):
        pred_answers = []
        for input_id, start_idx, end_idx in zip(
            batch.token_ids, start_logits.argmax(-1), end_logits.argmax(-1), strict=True
        ):
            pred_answers.append(batch.decode_tokens(input_id[start_idx : end_idx + 1]))

        return pred_answers

    def training_step(
        self, batch: TokenizedDocumentInstance, **kwargs
    ) -> "SequenceQAModelOutput":
        """
        Performs a single training step.

        Args:
            batch (TokenizedDocumentInstance): The input batch of data.
            **kwargs: Additional arguments.

        Returns:
            ClassificationModelOutput: The output of the training step, including loss and logits.
        """
        if self._training_overflow_strategy == OverflowStrategy.select_all:
            batch.select_all_overflow_samples()
        elif self._training_overflow_strategy == OverflowStrategy.select_random:
            batch.select_random_overflow_samples()
        elif self._training_overflow_strategy == OverflowStrategy.select_first:
            batch.select_first_overflow_samples()

        output = self._model_forward(batch)
        return SequenceQAModelOutput(
            loss=output.loss,
            start_logits=output.start_logits,
            end_logits=output.end_logits,
            words=batch.words,
            word_ids=batch.word_ids,
            sequence_ids=batch.sequence_ids,
            question_id=batch.qa_pair.id,
            gold_answers=batch.qa_pair.answer_text,
        )

    def evaluation_step(
        self, batch: TokenizedDocumentInstance, **kwargs
    ) -> "SequenceQAModelOutput":
        """
        Performs a single evaluation step.

        Args:
            batch (TokenizedDocumentInstance): The input batch of data.
            **kwargs: Additional arguments.

        Returns:
            ClassificationModelOutput: The output of the evaluation step, including loss and logits.
        """
        if self._evaluation_overflow_strategy == OverflowStrategy.select_all:
            batch.select_all_overflow_samples()
        elif self._evaluation_overflow_strategy == OverflowStrategy.select_random:
            batch.select_random_overflow_samples()
        elif self._evaluation_overflow_strategy == OverflowStrategy.select_first:
            batch.select_first_overflow_samples()

        output = self._model_forward(batch)
        predicted_answers = self._extract_predicted_answers(
            batch, output.start_logits, output.end_logits
        )
        output = SequenceQAModelOutput(
            loss=output.loss.detach().cpu(),
            start_logits=output.start_logits.detach().cpu(),
            end_logits=output.end_logits.detach().cpu(),
            words=batch.words,
            word_ids=batch.word_ids.detach().cpu(),
            sequence_ids=batch.sequence_ids.detach().cpu(),
            question_id=batch.qa_pair.id.detach().cpu(),
            gold_answers=batch.qa_pair.answer_text,
            predicted_answers=predicted_answers,
        )
        return output

    def predict_step(
        self, batch: TokenizedDocumentInstance, **kwargs
    ) -> "SequenceQAModelOutput":
        """
        Performs a single evaluation step.

        Args:
            batch (TokenizedDocumentInstance): The input batch of data.
            **kwargs: Additional arguments.

        Returns:
            ClassificationModelOutput: The output of the evaluation step, including loss and logits.
        """
        if self._evaluation_overflow_strategy == OverflowStrategy.select_all:
            batch.select_all_overflow_samples()
        elif self._evaluation_overflow_strategy == OverflowStrategy.select_random:
            batch.select_random_overflow_samples()
        elif self._evaluation_overflow_strategy == OverflowStrategy.select_first:
            batch.select_first_overflow_samples()

        output = self._model_forward(batch)
        return SequenceQAModelOutput(
            loss=output.loss,
            start_logits=output.start_logits,
            end_logits=output.end_logits,
            words=batch.words,
            word_ids=batch.word_ids,
            sequence_ids=batch.sequence_ids,
            question_id=batch.qa_pair.id,
            gold_answers=batch.qa_pair.answer_text,
        )

    def _prepare_build_kwargs(self) -> dict[str, dict[str, Any]] | dict[str, Any]:
        """
        Prepares keyword arguments for building the model.

        Returns:
            Dict[str, Dict[str, Any]] | Dict[str, Any]: The prepared keyword arguments.
        """
        return {}

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
            "start_positions": batch.qa_pair.tokenized_answer_starts,
            "end_positions": batch.qa_pair.tokenized_answer_ends,
            "token_type_ids": batch.token_type_ids,
            "attention_mask": batch.attention_mask,
            "bbox": batch.token_bboxes if self._use_bbox else None,
            "pixel_values": batch.image.content if self._use_image else None,
        }
        return self._model(**inputs)
