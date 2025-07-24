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

from __future__ import annotations

from typing import TYPE_CHECKING

from atria_core.logger.logger import get_logger

from atria_models.core.diffusers_model import DiffusersModel
from atria_models.core.local_model import LocalModel
from atria_models.pipelines.atria_model_pipeline import (
    AtriaModelPipeline,
    AtriaModelPipelineConfig,
)
from atria_models.registry import MODEL_PIPELINE
from atria_models.utilities.nn_modules import _unnormalize_image

if TYPE_CHECKING:
    import torch
    from atria_core.types import DocumentInstance, ImageInstance
    from ignite.engine import Engine

    from atria_models.data_types.outputs import ModelOutput

logger = get_logger(__name__)

SupportedBatchDataTypes = DocumentInstance | ImageInstance


class AutoEncodingPipelineConfig(AtriaModelPipelineConfig):
    model: DiffusersModel | LocalModel
    loss_type: str = "l2"


MODEL_PIPELINE.register("autoencoding")


class AutoEncodingPipeline(AtriaModelPipeline):
    def _build_model(self) -> torch.nn.Module:
        """
        Builds the model and initializes the loss function.

        Returns:
            torch.nn.Module: The constructed model.
        """

        import torch

        if self.config.loss_type == "l2":
            self._loss_fn = torch.nn.MSELoss()
        else:
            raise NotImplementedError(
                f"Loss type {self.config.loss_type} is not implemented. Supported loss types are: l2"
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

        from atria_models.data_types.outputs import AutoEncoderModelOutput

        reconstruction = self.model(batch.image.content)
        loss = self._loss_fn(input=reconstruction, target=batch.image.content)
        return AutoEncoderModelOutput(
            loss=loss, real=batch.image.content, reconstructed=reconstruction
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

        from atria_models.data_types.outputs import AutoEncoderModelOutput

        reconstruction = self.model(batch.image.content)
        loss = self._loss_fn(input=reconstruction, target=batch.image.content)
        return AutoEncoderModelOutput(
            loss=loss, real=batch.image.content, reconstructed=reconstruction
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

        from atria_models.data_types.outputs import AutoEncoderModelOutput

        reconstruction = self.model(batch.image.content)
        loss = self._loss_fn(input=reconstruction, target=batch.image.content)
        return AutoEncoderModelOutput(
            loss=loss, real=batch.image.content, reconstructed=reconstruction
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
        import math

        import ignite.distributed as idist

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
