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

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from atria_core.logger.logger import get_logger
from atria_core.types import SequenceQAModelOutput, TaskType

from atria_models.core.local_model import LocalModel
from atria_models.core.transformers_model import QuestionAnsweringModel
from atria_models.pipelines.atria_model_pipeline import (
    AtriaModelPipeline,
    AtriaModelPipelineConfig,
)
from atria_models.pipelines.utilities import OverflowStrategy
from atria_models.registry import MODEL_PIPELINE

if TYPE_CHECKING:
    import torch
    from atria_transforms.data_types import TokenizedDocumentInstance

logger = get_logger(__name__)


class QuestionAnsweringPipelineConfig(AtriaModelPipelineConfig):
    model: QuestionAnsweringModel | LocalModel
    use_bbox: bool = True
    use_image: bool = True
    training_overflow_strategy: OverflowStrategy = OverflowStrategy.select_random
    evaluation_overflow_strategy: OverflowStrategy = OverflowStrategy.select_all
    input_stride: int = 0


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
        {"/metric@metrics.sequence_anls": "sequence_anls"},
    ],
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

    __config_cls__ = QuestionAnsweringPipelineConfig
    __task_type__: TaskType = TaskType.visual_question_answering

    def can_be_evaluated(self, batch: TokenizedDocumentInstance) -> bool:
        """
        Check if the batch can be evaluated.

        Args:
            batch (TokenizedDocumentInstance): The input batch of data.

        Returns:
            bool: True if the batch can be evaluated, False otherwise.
        """
        return (
            batch.tokenized_answer_start is not None
            and batch.tokenized_answer_end is not None
        )

    def _extract_predicted_answers(
        self,
        batch: TokenizedDocumentInstance,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
    ):
        pred_answers = []
        logger.info(
            "Extracting predicted answers from logits.: %s, %s",
            start_logits.argmax(-1),
            end_logits.argmax(-1),
        )
        for input_id, start_idx, end_idx in zip(
            batch.token_ids, start_logits.argmax(-1), end_logits.argmax(-1), strict=True
        ):
            pred_answers.append(batch.decode_tokens(input_id[start_idx : end_idx + 1]))

        return pred_answers

    def training_step(
        self, batch: TokenizedDocumentInstance, **kwargs
    ) -> SequenceQAModelOutput:
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
        predicted_answers = self._extract_predicted_answers(
            batch, output.start_logits, output.end_logits
        )
        return SequenceQAModelOutput(
            loss=output.loss,
            start_logits=output.start_logits,
            end_logits=output.end_logits,
            words=batch.words,
            word_ids=batch.word_ids,
            sequence_ids=batch.sequence_ids,
            question_id=batch.qa_pair.id,
            gold_answers=batch.qa_pair.answer_text,
            predicted_answers=predicted_answers,
        )

    def evaluation_step(
        self, batch: TokenizedDocumentInstance, **kwargs
    ) -> SequenceQAModelOutput:
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
        predicted_answers = self._extract_predicted_answers(
            batch, output.start_logits, output.end_logits
        )
        logger.info("Evaluation step output: %s", predicted_answers)
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
    ) -> SequenceQAModelOutput:
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
        predicted_answers = self._extract_predicted_answers(
            batch, output.start_logits, output.end_logits
        )
        logger = get_logger(__name__)
        logger.info("Evaluation step output: %s", predicted_answers)
        return SequenceQAModelOutput(
            loss=output.loss,
            start_logits=output.start_logits,
            end_logits=output.end_logits,
            words=batch.words,
            word_ids=batch.word_ids,
            sequence_ids=batch.sequence_ids,
            question_id=batch.qa_pair.id,
            gold_answers=batch.qa_pair.answer_text,
            predicted_answers=predicted_answers,
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
            "start_positions": batch.tokenized_answer_start,
            "end_positions": batch.tokenized_answer_end,
            "token_type_ids": batch.token_type_ids,
            "attention_mask": batch.attention_mask,
            "bbox": batch.token_bboxes if self.config.use_bbox else None,
            "pixel_values": batch.image.content if self.config.use_image else None,
        }
        return self._model(**inputs)
