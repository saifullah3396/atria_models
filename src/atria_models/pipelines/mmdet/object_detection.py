from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from atria_core.logger.logger import get_logger
from atria_core.types import TaskType, TrainingStage
from pydantic import Field

from atria_models.core.atria_model import AtriaModel
from atria_models.data_types.outputs import MMDetEvaluationOutput, MMDetTrainingOutput
from atria_models.pipelines.atria_model_pipeline import (
    AtriaModelPipeline,
    AtriaModelPipelineConfig,
    RegistryConfig,
)
from atria_models.registry import MODEL_PIPELINE

if TYPE_CHECKING:
    from atria_core.types import TaskType, TrainingStage
    from atria_transforms.core.mmdet import MMDetInput
    from ignite.engine import Engine
    from pydantic import Field

    from atria_models.data_types.outputs import (
        MMDetEvaluationOutput,
        MMDetTrainingOutput,
    )


logger = get_logger(__name__)


@dataclass
class NMSConfig:
    type: str = "nms"
    iou_threshold: float = 0.5


@dataclass
class TestTimeAugmentationConfig:
    nms: NMSConfig = Field(default_factory=NMSConfig)
    max_per_img: int = 100


class ObjectDetectionPipelineConfig(AtriaModelPipelineConfig):
    model: AtriaModel
    requires_test_time_aug: bool = False
    test_time_aug_config: TestTimeAugmentationConfig | None = (
        TestTimeAugmentationConfig()
    )


@MODEL_PIPELINE.register(
    "layout_analysis",
    defaults=[
        "_self_",
        {"/model@model": "mmdet"},
        {
            "/data_transform@runtime_transforms.train": "document_instance_mmdet_transform/train"
        },
        {
            "/data_transform@runtime_transforms.evaluation": "document_instance_mmdet_transform/train"
        },
    ],
    metric_configs=[RegistryConfig(name="cocoeval")],
)
class ObjectDetectionPipeline(AtriaModelPipeline):
    """
    A pipeline for image classification tasks.

    This class extends the `ClassificationPipeline` to include support for mixup augmentation
    and advanced loss functions.

    Attributes:
        model (Union[AtriaModel, Dict[str, AtriaModel]]): The model or dictionary of models.
        checkpoint_configs (Optional[List[CheckpointConfig]]): List of checkpoint configurations.
        mixup_config (Optional[MixupConfig]): Configuration for mixup augmentation.
    """

    __config_cls__ = ObjectDetectionPipelineConfig
    ___task_type____: TaskType = TaskType.object_detection

    def _build_model(self):
        from mmdet.models.detectors import BaseDetector
        from mmdet.models.test_time_augs import DetTTAModel
        from mmengine.model.test_time_aug import BaseTTAModel

        torch_model = super()._build_model()
        if isinstance(torch_model, AtriaModel):
            # get the base detector model from the AtriaModel
            torch_model = torch_model._model

        assert isinstance(torch_model, BaseDetector | BaseTTAModel), (
            "Model must be an instance of mmdet BaseDetector"
        )

        if self.config.requires_test_time_augmentation:
            self._tta_model = DetTTAModel(
                tta_cfg=self.config.test_time_aug_config,
                module=torch_model,
                data_preprocessor=torch_model.data_preprocessor,
            )

        return torch_model

    def training_step(
        self, batch: MMDetInput, training_engine: Engine, **kwargs
    ) -> MMDetTrainingOutput:
        from mmdet.models.detectors import BaseDetector

        from atria_models.data_types.outputs import MMDetTrainingOutput

        assert isinstance(self._model, BaseDetector), (
            "Model must be an instance of mmdet BaseDetector"
        )
        batch = batch.model_dump()
        batch = self._model.data_preprocessor(batch, training=True)
        losses = self._model._run_forward(batch, mode="loss")
        loss, loss_dict = self._model.parse_losses(losses)
        self._tb_logger.writer.add_scalars(
            "losses", loss_dict, training_engine.state.iteration
        )
        return MMDetTrainingOutput(loss=loss)

    def evaluation_step(
        self, batch: MMDetInput, stage: TrainingStage, **kwargs
    ) -> MMDetEvaluationOutput:
        import torch
        from mmdet.models.detectors import BaseDetector

        from atria_models.data_types.outputs import MMDetEvaluationOutput

        assert isinstance(self._model, BaseDetector), (
            "Model must be an instance of mmdet BaseDetector"
        )
        batch = batch.model_dump()
        if self.config.requires_test_time_augmentation:
            det_data_samples = self._tta_model.test_step(batch)
            return MMDetEvaluationOutput(
                loss=torch.tensor(0, device=batch["inputs"][0].device),
                det_data_samples=det_data_samples,
            )
        else:
            if stage == TrainingStage.validation:
                det_data_samples = self._model.val_step(batch)
            elif stage == TrainingStage.test:
                det_data_samples = self._model.test_step(batch)
            return MMDetEvaluationOutput(
                loss=torch.tensor(0, device=batch["inputs"][0].device),
                det_data_samples=det_data_samples,
            )

    def predict_step(self, batch: MMDetInput, **kwargs) -> MMDetEvaluationOutput:
        import torch
        from mmdet.models.detectors import BaseDetector

        from atria_models.data_types.outputs import MMDetEvaluationOutput

        assert isinstance(self._model, BaseDetector), (
            "Model must be an instance of mmdet BaseDetector"
        )
        batch = batch.model_dump()
        if self.config.requires_test_time_augmentation:
            det_data_samples = self._tta_model.test_step(batch)
            return MMDetEvaluationOutput(
                loss=torch.tensor(0, device=batch["inputs"][0].device),
                det_data_samples=det_data_samples,
            )
        else:
            det_data_samples = self._model.test_step(batch)
            return MMDetEvaluationOutput(
                loss=torch.tensor(0, device=batch["inputs"][0].device),
                det_data_samples=det_data_samples,
            )

    def visualization_step(
        self, batch: MMDetInput, training_engine: Engine, **kwargs
    ) -> None:
        import mmcv
        from mmdet.models.detectors import BaseDetector
        from mmdet.structures.bbox import scale_boxes
        from mmdet.visualization import DetLocalVisualizer

        assert isinstance(self._model, BaseDetector), (
            "Model must be an instance of mmdet BaseDetector"
        )
        batch = batch.model_dump()
        if self.config.requires_test_time_augmentation:
            det_data_samples = self._tta_model.test_step(batch)
        else:
            det_data_samples = self._model.test_step(batch)

        # Determine output directory
        output_dir = (
            Path(self._tb_logger.logdir).parent
            / f"visualizations_{training_engine.state.epoch}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        visualizer = DetLocalVisualizer()
        visualizer.dataset_meta["classes"] = [
            label.upper() for label in self._dataset_metadata.dataset_labels.layout
        ]

        logger.info(f"Saving visualizations to {output_dir}")
        for idx, data_sample in enumerate(det_data_samples):
            if (
                self.config.requires_test_time_augmentation
            ):  # for tta, the inputs are of shape (B, TTA, C, H, W)
                img = batch["inputs"][0][idx].cpu().numpy().transpose(1, 2, 0)
            else:  # else the inputs are of shape (B, C, H, W)
                img = batch["inputs"][idx].cpu().numpy().transpose(1, 2, 0)

            scale_factor = data_sample.metainfo.get("scale_factor")
            if "pred_instances" in data_sample:
                data_sample.pred_instances.bboxes = scale_boxes(
                    data_sample.pred_instances.bboxes, scale_factor
                )

            # Draw predictions
            visualizer.add_datasample(
                name=f"sample_{idx}",
                image=img,
                data_sample=data_sample,
                draw_gt=True,
                draw_pred=True,
            )

            # Save visualization
            output_path = output_dir / f"sample_{idx}.png"
            logger.info(f"Saving visualization to {output_path}")
            mmcv.imwrite(visualizer.get_image(), output_path)
