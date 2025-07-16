"""
AutoEncoding Pipeline Module

This module defines the `AutoEncodingPipeline` class, which is a specific implementation
of the `AtriaModelPipeline` for autoencoding tasks. It provides methods for training,
evaluation, prediction, and visualization of autoencoding models.

Classes:
    - AutoEncodingPipeline: A pipeline for autoencoding tasks.

Dependencies:
    - ignite: For distributed training and logging utilities.
    - torch: For PyTorch operations.
    - atria_core.logger: For logging utilities.
    - atria_models.utilities: For checkpoint management and neural network utilities.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

import math
from functools import partial

import ignite.distributed as idist
import torch
from atria_core.logger.logger import get_logger
from atria_core.types import DatasetMetadata, DocumentInstance, ImageInstance
from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Engine

from atria_models.core.atria_model import AtriaModel
from atria_models.outputs import AutoEncoderModelOutput, ModelOutput
from atria_models.pipelines.atria_model_pipeline import AtriaModelPipeline
from atria_models.utilities.checkpoints import CheckpointConfig
from atria_models.utilities.nn_modules import _unnormalize_image

logger = get_logger(__name__)

SupportedBatchDataTypes = DocumentInstance | ImageInstance


class AutoEncodingPipeline(AtriaModelPipeline):
    """
    AutoEncodingPipeline is a specific implementation of the AtriaModelPipeline for autoencoding tasks.

    Args:
        model_factory (Union[partial[AtriaModel], Dict[str, partial[AtriaModel]]]): Factory for creating the model.
        dataset_metadata (Optional[DatasetMetadata]): Metadata for the dataset.
        checkpoint_configs (Optional[List[CheckpointConfig]]): Configuration for model checkpoints.
        tb_logger (Optional[TensorboardLogger]): Tensorboard logger for visualization.
        loss_type (str): Type of loss function to use. Default is "l2".

    Attributes:
        _loss_type (str): Type of loss function.
        _loss_fn (torch.nn.Module): Loss function module.
    """

    def __init__(
        self,
        model_factory: partial[AtriaModel] | dict[str, partial[AtriaModel]],
        dataset_metadata: DatasetMetadata | None = None,
        checkpoint_configs: list[CheckpointConfig] | None = None,
        tb_logger: TensorboardLogger | None = None,
        loss_type: str = "l2",
    ):
        """
        Initializes the AutoEncodingPipeline.

        Args:
            model_factory (Union[partial[AtriaModel], Dict[str, partial[AtriaModel]]]): Factory for creating the model.
            dataset_metadata (Optional[DatasetMetadata]): Metadata for the dataset.
            checkpoint_configs (Optional[List[CheckpointConfig]]): Configuration for model checkpoints.
            tb_logger (Optional[TensorboardLogger]): Tensorboard logger for visualization.
            loss_type (str): Type of loss function to use. Default is "l2".
        """
        self._loss_type = loss_type

        super().__init__(
            model=model_factory,
            dataset_metadata=dataset_metadata,
            checkpoint_configs=checkpoint_configs,
            tb_logger=tb_logger,
        )

    def _build_model(self) -> torch.nn.Module:
        """
        Builds the model and initializes the loss function.

        Returns:
            torch.nn.Module: The constructed model.
        """
        if self._loss_type == "l2":
            self._loss_fn = torch.nn.MSELoss()
        else:
            raise NotImplementedError(
                f"Loss type {self._loss_type} is not implemented. Supported loss types are: l2"
            )

        return super()._build_model()

    def training_step(self, batch: SupportedBatchDataTypes, **kwargs) -> ModelOutput:
        """
        Performs a single training step.

        Args:
            batch (SupportedBatchDataTypes): Batch of data for training.
            **kwargs: Additional arguments.

        Returns:
            ModelOutput: Output containing loss, real input, and reconstructed output.
        """
        reconstruction = self.model(batch.image.content)
        loss = self._loss_fn(input=reconstruction, target=input)
        return AutoEncoderModelOutput(
            loss=loss, real=input, reconstructed=reconstruction
        )

    def evaluation_step(self, batch: SupportedBatchDataTypes, **kwargs) -> ModelOutput:
        """
        Performs a single evaluation step.

        Args:
            batch (SupportedBatchDataTypes): Batch of data for evaluation.
            **kwargs: Additional arguments.

        Returns:
            ModelOutput: Output containing loss, real input, and reconstructed output.
        """
        reconstruction = self.model(batch.image.content)
        loss = self._loss_fn(input=reconstruction, target=input)
        return AutoEncoderModelOutput(
            loss=loss, real=input, reconstructed=reconstruction
        )

    def predict_step(self, batch: SupportedBatchDataTypes, **kwargs) -> ModelOutput:
        """
        Performs a single prediction step.

        Args:
            batch (SupportedBatchDataTypes): Batch of data for prediction.
            **kwargs: Additional arguments.

        Returns:
            ModelOutput: Output containing loss, real input, and reconstructed output.
        """
        reconstruction = self.model(batch.image.content)
        loss = self._loss_fn(input=reconstruction, target=input)
        return AutoEncoderModelOutput(
            loss=loss, real=input, reconstructed=reconstruction
        )

    def visualization_step(
        self,
        batch: SupportedBatchDataTypes,
        evaluation_engine: Engine | None = None,
        training_engine: Engine | None = None,
        **kwargs,
    ) -> None:
        """
        Visualizes the input and reconstructed images using Tensorboard.

        Args:
            batch (SupportedBatchDataTypes): Batch of data for visualization.
            evaluation_engine (Optional[Engine]): Engine for evaluation.
            training_engine (Optional[Engine]): Engine for training.
            **kwargs: Additional arguments.
        """
        reconstruction = self.model(batch.image.content)
        global_step = (
            training_engine.state.iteration if training_engine is not None else 1
        )
        if idist.get_rank() == 0:
            import torchvision

            # this only saves first batch always if you want you can shuffle validation set and save random batches
            logger.info(
                f"Saving image batch {evaluation_engine.state.iteration} to tensorboard"
            )
            input = batch.image.content

            if input.min() < 0 and input.max() > 1.0:
                # assume image is normalized -1 to 1 here
                input = _unnormalize_image(input)
                reconstruction = _unnormalize_image(reconstruction)

            # save images to tensorboard
            num_samples = input.shape[0]
            self._tb_logger.writer.add_image(
                "visualization/input",
                torchvision.utils.make_grid(input, nrow=int(math.sqrt(num_samples))),
                global_step,
            )
            self._tb_logger.writer.add_image(
                "visualization/reconstruction",
                torchvision.utils.make_grid(
                    reconstruction, nrow=int(math.sqrt(num_samples))
                ),
                global_step,  # this is iteration of the training engine1
            )
