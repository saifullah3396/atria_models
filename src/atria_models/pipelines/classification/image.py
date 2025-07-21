"""
Image Classification Pipeline Module

This module defines the `ImageClassificationPipeline` class, which is a specific implementation
of the `ClassificationPipeline` for image classification tasks. It includes support for mixup
augmentation and advanced loss functions.

Classes:
    - ImageClassificationPipeline: A pipeline for image classification tasks.
    - MixupConfig: Configuration for mixup augmentation.

Dependencies:
    - torch: For PyTorch operations.
    - timm: For mixup augmentation and loss functions.
    - atria_models.utilities: For checkpoint management and neural network utilities.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Union

from atria_core.transforms import DataTransformsDict
from atria_core.types import TaskType
from atria_models.core.atria_model import AtriaModel
from atria_models.core.timm_model import TimmModel
from atria_models.core.torchvision_model import TorchHubModel
from atria_models.core.transformers_model import ImageClassificationModel
from atria_models.data_types.outputs import ClassificationModelOutput
from atria_models.pipelines.atria_model_pipeline import MetricInitializer
from atria_models.pipelines.classification.base import ClassificationPipeline
from atria_models.registry import MODEL_PIPELINE
from atria_models.utilities.checkpoints import CheckpointConfig
from atria_models.utilities.nn_modules import AtriaModelDict, _get_logits_from_output

if TYPE_CHECKING:
    from atria_core.types import DocumentInstance, ImageInstance

SupportedBatchDataTypes = Union["DocumentInstance", "ImageInstance"]


@dataclass
class MixupConfig:
    """
    Configuration for mixup augmentation.

    Attributes:
        mixup_alpha (float): Mixup interpolation coefficient.
        cutmix_alpha (float): CutMix interpolation coefficient.
        cutmix_minmax (Optional[float]): Minimum and maximum bounds for CutMix.
        prob (float): Probability of applying mixup or CutMix.
        switch_prob (float): Probability of switching between mixup and CutMix.
        mode (str): Mode of operation, e.g., 'batch'.
        mixup_prob (float): Probability of applying mixup.
        correct_lam (bool): Whether to correct lambda values.
        label_smoothing (float): Label smoothing factor.
    """

    mixup_alpha: float = 1.0
    cutmix_alpha: float = 0.0
    cutmix_minmax: float | None = None
    prob: float = 1.0
    switch_prob: float = 0.5
    mode: str = "batch"
    mixup_prob: float = 1.0
    correct_lam: bool = True
    label_smoothing: float = 0.1


@MODEL_PIPELINE.register(
    "image_classification",
    hydra_defaults=[
        "_self_",
        {"/model@model": "timm"},
        {"/data_transform@runtime_transforms.train": "image/default"},
        {"/data_transform@runtime_transforms.evaluation": "image/default"},
    ],
    metrics=[
        MetricInitializer(name="accuracy"),
        MetricInitializer(name="precision"),
        MetricInitializer(name="recall"),
        MetricInitializer(name="f1_score"),
        MetricInitializer(name="confusion_matrix"),
    ],
)
class ImageClassificationPipeline(ClassificationPipeline):
    """
    A pipeline for image classification tasks.

    This class extends the `ClassificationPipeline` to include support for mixup augmentation
    and advanced loss functions.

    Attributes:
        model (Union[AtriaModel, Dict[str, AtriaModel]]): The model or dictionary of models.
        checkpoint_configs (Optional[List[CheckpointConfig]]): List of checkpoint configurations.
        mixup_config (Optional[MixupConfig]): Configuration for mixup augmentation.
        runtime_transforms (Optional[DataTransformsDict]): Runtime data transformations.
        metric_factory (Optional[Dict[str, Callable]]): Factory for metrics.
    """

    _TASK_TYPE: TaskType = TaskType.image_classification

    def __init__(
        self,
        model: TimmModel | TorchHubModel | ImageClassificationModel,
        checkpoint_configs: list[CheckpointConfig] | None = None,
        mixup_config: MixupConfig | None = None,
        metrics: list[MetricInitializer] | None = None,
        runtime_transforms: DataTransformsDict = DataTransformsDict(),
    ):
        """
        Initialize the ImageClassificationPipeline.

        Args:
            model (Union[AtriaModel, Dict[str, AtriaModel]]): The model or dictionary of models.
            checkpoint_configs (Optional[List[CheckpointConfig]]): List of checkpoint configurations.
            mixup_config (Optional[MixupConfig]): Configuration for mixup augmentation.
            runtime_transforms (Optional[DataTransformsDict]): Runtime data transformations.
            metric_factory (Optional[Dict[str, Callable]]): Factory for metrics.
        """
        self._mixup_config = mixup_config

        super().__init__(
            model=model,
            checkpoint_configs=checkpoint_configs,
            metrics=metrics,
            runtime_transforms=runtime_transforms,
        )

    def training_step(
        self, batch: SupportedBatchDataTypes, **kwargs
    ) -> "ClassificationModelOutput":
        """
        Perform a single training step.

        Args:
            batch (ImageInstance): The input batch of data.
            **kwargs: Additional arguments.

        Returns:
            ClassificationModelOutput: The output containing loss, logits, and labels.
        """
        from atria_models.data_types.outputs import ClassificationModelOutput

        if self._mixup is not None:
            batch.image.content, batch.gt.classification.label.value = self._mixup(
                batch.image.content, batch.gt.classification.label.value
            )
        logits = _get_logits_from_output(self._model_forward(batch))
        loss = self._loss_fn_train(logits, batch.gt.classification.label.value)
        return ClassificationModelOutput(
            loss=loss, logits=logits, label=batch.gt.classification.label
        )

    def _build_model(self) -> AtriaModel | AtriaModelDict:
        """
        Build the model and configure loss functions.

        Returns:
            Union[AtriaModel, AtriaModelDict]: The built model or dictionary of models.
        """
        from timm.data.mixup import Mixup
        from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
        from torch import nn

        self._mixup: Mixup = None
        if self._mixup_config is not None:
            self._mixup = Mixup(
                num_classes=len(self._dataset_metadata.dataset_labels.classification),
                mixup_alpha=self._mixup_config.mixup_alpha,
                cutmix_alpha=self._mixup_config.cutmix_alpha,
                cutmix_minmax=self._mixup_config.cutmix_minmax,
                label_smoothing=self._mixup_config.label_smoothing,
            )
        if self._mixup is not None:
            self._loss_fn_train = (
                LabelSmoothingCrossEntropy(self._mixup.label_smoothing)
                if self._mixup.label_smoothing > 0.0
                else SoftTargetCrossEntropy()
            )
        else:
            self._loss_fn_train = nn.CrossEntropyLoss()
        self._loss_fn_eval = nn.CrossEntropyLoss()

        return super()._build_model()

    def _model_forward(self, batch: Union["ImageInstance", "DocumentInstance"]) -> Any:
        """
        Forward pass through the model.

        Args:
            batch (ImageInstance): The input batch of data.

        Returns:
            Any: The model output.
        """
        return self._model(batch.image.content)
