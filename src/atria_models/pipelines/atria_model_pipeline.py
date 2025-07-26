"""
Atria Task Pipeline Module

This module defines the `AtriaTaskModule` class, which serves as an abstract base class
for constructing and managing task-specific PyTorch models. It provides functionality
for building models, managing checkpoints, handling distributed training, and defining
task-specific forward passes.

Classes:
    - AtriaTaskModule: Abstract base class for task-specific PyTorch models.

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

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from functools import cached_property, wraps
from typing import TYPE_CHECKING, Any, ClassVar

from atria_core.logger import get_logger
from atria_core.transforms.base import DataTransformsDict
from atria_core.types import TrainingStage
from atria_metrics import MetricBuilder
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict

from atria_models.core.atria_model import AtriaModel
from atria_models.utilities.checkpoints import CheckpointConfig

if TYPE_CHECKING:
    import torch
    from atria_core.types import (
        BaseDataInstance,
        DatasetMetadata,
        ModelOutput,
        TaskType,
    )
    from ignite.contrib.handlers import TensorboardLogger
    from ignite.engine import Engine
    from ignite.handlers import ProgressBar

    from atria_models.utilities.nn_modules import AtriaModelDict

logger = get_logger(__name__)


class AtriaModelPipelineConfig(BaseModel):
    """
    Configuration model for AtriaModelPipeline.

    This model is used to define the configuration parameters for the AtriaModelPipeline.
    It includes fields for model, checkpoint configurations, metric configurations, and runtime transforms.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str | None = None
    model: AtriaModel | dict[str, AtriaModel]
    checkpoint_configs: list[CheckpointConfig] | None = None
    metric_builders: dict[str, MetricBuilder] | None = None
    runtime_transforms: DataTransformsDict = DataTransformsDict()


class ModelConfigMixin:
    __config_cls__: type[AtriaModelPipelineConfig]

    def __init__(self, **kwargs):
        config_cls = getattr(self.__class__, "__config_cls__", None)
        assert issubclass(config_cls, AtriaModelPipelineConfig), (
            f"{self.__class__.__name__} must define a __config_cls__ attribute "
            "that is a subclass of ModelPipelineConfig."
        )
        self._config = config_cls(**kwargs)
        self._build_config = None
        super().__init__()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Validate presence of Config at class definition time
        if not hasattr(cls, "__config_cls__"):
            raise TypeError(
                f"{cls.__name__} must define a nested `__config_cls__` class."
            )

        if not issubclass(cls.__config_cls__, AtriaModelPipelineConfig):
            raise TypeError(
                f"{cls.__name__}.Config must subclass pydantic.ModelPipelineConfig. Got {cls.__config_cls__} instead."
            )

    def prepare_build_config(self):
        from hydra_zen import builds

        if self.__config_cls__ is None:
            raise TypeError(
                f"{self.__class__.__name__} must define a __config_cls__ attribute."
            )

        self._build_config = OmegaConf.to_container(
            OmegaConf.create(
                builds(
                    self.__class__,
                    populate_full_signature=True,
                    model={
                        key: value.build_config
                        for key, value in self._config.model.items()
                    }
                    if isinstance(self._config.model, dict)
                    else self._config.model.build_config,
                    checkpoint_configs=self._config.checkpoint_configs
                    if self._config.checkpoint_configs is not None
                    else None,
                    metric_builders=self._config.metric_builders
                    if self._config.metric_builders is not None
                    else None,
                    runtime_transforms=self._config.runtime_transforms.build_config
                    if self._config.runtime_transforms is not None
                    else None,
                )
            )
        )

    @cached_property
    def config(self) -> AtriaModelPipelineConfig:
        return self._config

    @cached_property
    def build_config(self) -> dict:
        if self._build_config is None:
            self.prepare_build_config()
        return self._build_config

    @cached_property
    def config_hash(self) -> str:
        """
        Hash of the dataset configuration for versioning.

        Returns:
            8-character hash string based on configuration content
        """
        import hashlib
        import json

        return hashlib.sha256(
            json.dumps(self.build_config, sort_keys=True).encode()
        ).hexdigest()[:8]


class AtriaModelPipeline(ABC, ModelConfigMixin):
    """
    Abstract base class for task-specific PyTorch models.

    This class provides functionality for building models, managing checkpoints,
    handling distributed training, and defining task-specific forward passes.

    Attributes:
        _REQUIRES_MODEL_DICT (bool): Whether the model builder must be a dictionary.
        _REQUIRED_MODEL_KEYS (List[str]): Required keys for the model builder dictionary.
    """

    __config_cls__ = AtriaModelPipelineConfig
    __requires_model_dict__: ClassVar[bool] = False
    __required_model_dict_keys__: ClassVar[list[str]] = []
    __task_type__: TaskType

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        self._dataset_metadata = None
        self._tb_logger = None
        self._progress_bar = None
        self._is_built = False
        self._apply_runtime_transforms = False
        self._model: torch.nn.Module | AtriaModelDict = None
        self._validate_model_config()

    def __init_subclass__(cls):
        super().__init_subclass__()

        original_training_step = cls.training_step
        original_evaluation_step = cls.evaluation_step
        original_predict_step = cls.predict_step
        original_visualization_step = cls.visualization_step

        @wraps(original_training_step)
        def wrapped_training_step(
            self: AtriaModelPipeline,
            batch: BaseDataInstance | list[BaseDataInstance],
            **kwargs,
        ):
            batch = self._transform_batch(batch, transform_type="evaluation")
            return original_training_step(self, batch, **kwargs)

        @wraps(original_evaluation_step)
        def wrapped_evaluation_step(
            self: AtriaModelPipeline,
            batch: BaseDataInstance | list[BaseDataInstance],
            **kwargs,
        ):
            batch = self._transform_batch(batch, transform_type="evaluation")
            return original_evaluation_step(self, batch, **kwargs)

        @wraps(original_predict_step)
        def wrapped_predict_step(
            self: AtriaModelPipeline,
            batch: BaseDataInstance | list[BaseDataInstance],
            **kwargs,
        ):
            batch = self._transform_batch(batch, transform_type="evaluation")
            return original_predict_step(self, batch, **kwargs)

        @wraps(original_visualization_step)
        def wrapped_visualization_step(
            self: AtriaModelPipeline,
            batch: BaseDataInstance | list[BaseDataInstance],
            **kwargs,
        ):
            batch = self._transform_batch(batch, transform_type="evaluation")
            return original_visualization_step(self, batch, **kwargs)

        cls.training_step = wrapped_training_step
        cls.evaluation_step = wrapped_evaluation_step
        cls.predict_step = wrapped_predict_step
        cls.visualization_step = wrapped_visualization_step

    @property
    def task_type(self) -> TaskType:
        """
        Returns the task type of the model pipeline

        Returns:
            TaskType: The task type.
        """
        return self.__task_type__

    @property
    def model_name(self) -> str:
        """
        Returns the name of the model.

        Returns:
            str: The model name.
        """
        if isinstance(self.config.model, dict):
            return ",".join([cfg.model_name for cfg in self.config.model.values()])
        return self.config.model.model_name

    @property
    def pipeline_name(self) -> str:
        """
        Returns the name of the task module.

        Returns:
            str: The task module name.
        """
        return self.__class__.__name__

    @property
    def model(self) -> torch.nn.Module | AtriaModelDict:
        """
        Returns the underlying PyTorch model.

        Returns:
            Union[torch.nn.Module, AtriaModelDict]: The PyTorch model.
        """
        return self._model

    @property
    def metrics(self) -> dict[str, Callable]:
        """
        Returns the metrics for the model.

        Returns:
            dict[str, Callable]: A dictionary of metrics.
        """
        return self._metrics

    @property
    def ema_modules(self) -> torch.nn.Module | dict[str, torch.nn.Module]:
        """
        Returns the Exponential Moving Average (EMA) modules.

        Returns:
            Union[torch.nn.Module, Dict[str, torch.nn.Module]]: The EMA modules.
        """
        from atria_models.utilities.nn_modules import AtriaModelDict

        return (
            self._model.trainable_models
            if isinstance(self._model, AtriaModelDict)
            else self._model
        )

    @property
    def progress_bar(self) -> ProgressBar:
        """
        Returns the progress bar for the model.

        Returns:
            ProgressBar: The progress bar instance.
        """
        return self._progress_bar

    @progress_bar.setter
    def progress_bar(self, progress_bar: ProgressBar) -> None:
        """
        Sets the progress bar for the model.

        Args:
            progress_bar (ProgressBar): The progress bar instance.
        """
        self._progress_bar = progress_bar

    def enable_runtime_transforms(self) -> None:
        """
        Enables the application of runtime transforms during model execution.
        """
        self._apply_runtime_transforms = True

    def disable_runtime_transforms(self) -> None:
        """
        Disables the application of runtime transforms during model execution.
        """
        self._apply_runtime_transforms = False

    def build(
        self,
        dataset_metadata: DatasetMetadata | None,
        tb_logger: TensorboardLogger | None = None,
    ) -> None:
        """
        Builds the model using the provided dataset metadata and Tensorboard logger.

        Args:
            dataset_metadata (DatasetMetadata): Metadata for the dataset.
            tb_logger (Optional[TensorboardLogger]): Tensorboard logger instance.
        """
        import ignite.distributed as idist

        from atria_models.utilities.checkpoints import CheckpointManager
        from atria_models.utilities.nn_modules import _validate_built_model

        self._dataset_metadata = dataset_metadata
        self._tb_logger = tb_logger

        if idist.get_rank() > 0:  # Stop all ranks > 0
            idist.barrier()

        self._model = _validate_built_model(self._build_model())
        self.prepare_build_config()

        if self.config.checkpoint_configs is not None:
            CheckpointManager.load_checkpoints(
                model=self._model, checkpoint_configs=self.config.checkpoint_configs
            )

        if idist.get_rank() == 0:
            idist.barrier()

        self._is_built = True

        return self

    def build_from_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """
        Builds the model from a checkpoint.

        Args:
            checkpoint (dict[str, Any]): The checkpoint dictionary containing model state.
        """
        from atria_core.types import DatasetMetadata

        dataset_metadata = checkpoint.pop("dataset_metadata", None)
        if dataset_metadata is None:
            raise ValueError(
                "Checkpoint must contain 'dataset_metadata'. "
                "Please ensure the model was saved with the dataset metadata."
            )
        dataset_metadata = DatasetMetadata.model_validate(dataset_metadata)
        self.build(
            dataset_metadata=dataset_metadata,
            tb_logger=None,  # Tensorboard logger is not used in this context
        )
        self.load_state_dict(checkpoint)
        return self

    def to_device(self, device: str | torch.device, sync_bn: bool = False) -> None:
        from atria_models.utilities.nn_modules import AtriaModelDict, _module_to_device

        if isinstance(self._model, AtriaModelDict):
            self._model.trainable_models = _module_to_device(
                self._model.trainable_models,
                device=device,
                sync_bn=sync_bn,
                prepare_for_distributed=True,
            )
            self._model.non_trainable_models = _module_to_device(
                self._model.non_trainable_models,
                device=device,
                sync_bn=sync_bn,
                prepare_for_distributed=False,
            )
        else:
            self._model = _module_to_device(
                self._model,
                device=device,
                sync_bn=sync_bn,
                prepare_for_distributed=True,
            )

        self._metrics = self._build_metrics(device=device)

        return self

    def train(self):
        """
        Sets the model to training mode.

        Returns:
            self: The current instance.
        """
        from atria_models.utilities.nn_modules import AtriaModelDict

        if isinstance(self._model, AtriaModelDict):
            self._model.trainable_models.train()
        else:
            self._model.train()

        return self

    def eval(self):
        """
        Sets the model to evaluation mode.

        Returns:
            self: The current instance.
        """
        from atria_models.utilities.nn_modules import AtriaModelDict

        if isinstance(self._model, AtriaModelDict):
            self._model.trainable_models.eval()
            self._model.non_trainable_models.eval()
        else:
            self._model.eval()

        return self

    def half(self):
        """
        Converts the model to half-precision mode.

        Returns:
            self: The current instance.
        """
        from atria_models.utilities.nn_modules import AtriaModelDict

        if isinstance(self._model, AtriaModelDict):
            self._model.trainable_models.half()
            self._model.non_trainable_models.half()
        else:
            self._model.half()

        return self

    def optimizer_parameters(self) -> Mapping[str, list[torch.nn.Parameter]]:
        """
        Retrieves the optimizer parameters for the model.

        Returns:
            Mapping[str, List[torch.nn.Parameter]]: A mapping of parameter groups.
        """
        import torch
        from atria_core.constants import _DEFAULT_OPTIMIZER_PARAMETERS_KEY

        from atria_models.utilities.nn_modules import AtriaModelDict

        if isinstance(self._model, AtriaModelDict):
            return {
                _DEFAULT_OPTIMIZER_PARAMETERS_KEY: list(
                    self._model.trainable_models.parameters()
                )
            }
        elif isinstance(self._model, torch.nn.Module):
            return {_DEFAULT_OPTIMIZER_PARAMETERS_KEY: list(self._model.parameters())}
        else:
            raise ValueError(
                f"Model must be a torch nn.Module or a dictionary of torch nn.Modules. Got {type(self._model)}"
            )

    def state_dict(self):
        """
        Retrieves the state dictionary of the model.

        Returns:
            dict: The state dictionary.
        """

        from atria_models.utilities.nn_modules import AtriaModelDict

        state_dict = {}
        if self._dataset_metadata is not None:
            state_dict["dataset_metadata"] = self._dataset_metadata.state_dict()
        if isinstance(self.model, AtriaModelDict):
            state_dict["trainable_models"] = self.model.trainable_models.state_dict()
            state_dict["non_trainable_models"] = (
                self.model.non_trainable_models.state_dict()
            )
        else:
            state_dict["model"] = self.model.state_dict()
        state_dict["config"] = self.build_config
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        """
        Loads the state dictionary into the model.

        Args:
            state_dict (dict): The state dictionary to load.
        """
        if "dataset_metadata" in state_dict:
            self._dataset_metadata.load_state_dict(state_dict["dataset_metadata"])
        if "trainable_models" in state_dict:
            assert "non_trainable_models" in state_dict, (
                "Both trainable and non-trainable models must be present in the state dict"
            )
            self._model.trainable_models.load_state_dict(state_dict["trainable_models"])
            self._model.non_trainable_models.load_state_dict(
                state_dict["non_trainable_models"]
            )
        elif "model" in state_dict:
            if "_model" in state_dict["model"]:
                self._model._model.load_state_dict(state_dict["model"]["_model"])
            else:
                self._model.load_state_dict(state_dict["model"])

    def upload_to_hub(
        self,
        name: str,
        branch: str = "main",
        description: str | None = None,
        is_public: bool = False,
    ) -> None:
        from atria_models.utilities.checkpoints import _checkpoint_to_bytes

        assert self._is_built, (
            "Model must be built before uploading to the hub. "
            "Call `build()` method before uploading."
        )
        try:
            import atria_hub  # noqa: F401
        except ImportError:
            raise ImportError(
                "The 'atria_hub' package is required to load datasets from the hub. "
                "Please install it using 'uv add https://github.com/saifullah3396/atria_hub'."
            )

        from atria_hub.hub import AtriaHub

        if description is None:
            description = f"A {self.__class__.__name__} model checkpoint."

        logger.info(
            f"Uploading model {self.__class__.__name__} to hub with name {name} and config {branch}."
        )

        # initialize the AtriaHub client
        hub = AtriaHub().initialize()

        # create a new model in the hub or get the existing one
        model = hub.models.get_or_create(
            username=hub.auth.username,
            name=name,
            description=description,
            task_type=self.task_type,
            is_public=is_public,
        )

        # upload the model checkpoint to the hub at the specified branch
        hub.models.upload_checkpoint(
            model=model,
            branch=branch,
            model_checkpoint=_checkpoint_to_bytes(self.state_dict()),
            model_config=self.config,
        )

    @classmethod
    def load_from_registry(
        cls,
        pipeline_name: str,
        model: str,
        model_name: str,
        dataset_metadata: DatasetMetadata,
        provider: str | None = None,
        overrides: list[str] | None = None,
        search_pkgs: list[str] | None = None,
        tb_logger: TensorboardLogger | None = None,
    ) -> AtriaModelPipeline:
        """
        Loads a model pipeline from the registry.

        Args:
            name (str): The name of the model pipeline.
            branch (str): The branch to load the model from.
            override_config (dict | None): Optional configuration overrides.

        Returns:
            AtriaModelPipeline: The loaded model pipeline instance.
        """
        from atria_models.registry import MODEL_PIPELINE

        overrides = overrides or []
        if search_pkgs is None:
            search_pkgs = ["atria_transforms"]
        model_pipeline: AtriaModelPipeline = MODEL_PIPELINE.load_from_registry(
            pipeline_name,
            provider=provider,
            overrides=[
                f"model@model_pipeline.model={model}",
                f"model_pipeline.model.model_name={model_name}",
            ]
            + overrides,
            search_pkgs=search_pkgs,
        )
        return model_pipeline.build(
            dataset_metadata=dataset_metadata, tb_logger=tb_logger
        )

    @classmethod
    def _validate_model_name(cls, name: str) -> tuple[str, str, str | None]:
        if "/" not in name:
            raise ValueError(
                f"Invalid model name format: {name}. "
                "Expected format is 'username/model_name' or 'username/model_name/branch'."
            )

        parts = name.split("/")
        if len(parts) == 2:
            return parts[0], parts[1], None
        elif len(parts) == 3:
            return parts[0], parts[1], parts[2]
        else:
            raise ValueError(
                f"Invalid model name format: {name}. "
                "Expected format is 'username/model_name' or 'username/model_name/branch'."
            )

    @classmethod
    def load_from_hub(
        cls, name: str, override_config: dict | None = None
    ) -> AtriaModelPipeline:
        from atria_registry.utilities import _instantiate_object_from_config

        from atria_models.utilities.checkpoints import _bytes_to_checkpoint

        try:
            import atria_hub  # noqa: F401
        except ImportError:
            raise ImportError(
                "The 'atria_hub' package is required to load models from the hub. "
                "Please install it using 'uv add https://github.com/saifullah3396/atria_hub'."
            )

        from atria_hub.hub import AtriaHub

        # validate the model name format
        username, name, branch = cls._validate_model_name(name)

        logger.info(f"Loading model {username}/{name}/{branch} from hub.")

        # initialize the AtriaHub client
        hub = AtriaHub().initialize()

        # create a new model in the hub or get the existing one
        model = hub.models.get_by_name(username=username, name=name)

        # upload the model checkpoint to the hub at the specified branch
        checkpoint, config = hub.models.load_checkpoint_and_config(
            model_repo_id=model.repo_id, branch=branch
        )
        checkpoint = _bytes_to_checkpoint(checkpoint)

        # first we get the model config
        config = checkpoint.pop("config", None)
        if config is None:
            raise ValueError(
                "The model checkpoint does not contain a 'config' key. "
                "Please ensure the model was saved with the configuration."
            )
        model: AtriaModelPipeline = _instantiate_object_from_config(
            config, override_config
        )
        return model.build_from_checkpoint(checkpoint)

    def _transform_batch(
        self,
        batch: BaseDataInstance | list[BaseDataInstance],
        transform_type: str = "train",
    ) -> BaseDataInstance:
        """
        Hook to be called before the training step.
        Override this method in subclasses to implement custom behavior.
        """
        if self._apply_runtime_transforms:
            transform = (
                self.config.runtime_transforms.train
                if transform_type == "train"
                else self.config.runtime_transforms.evaluation
            )
            if transform is None:
                raise ValueError(
                    "You have enabled runtime transforms, but no transform is defined "
                    f"for {transform_type} type. Please define a transform in the model pipeline."
                )
            assert isinstance(batch, list) and not any(x.is_batched for x in batch), (
                "Batch must be a list of untransformed BaseDataInstance when applying runtime transforms."
            )
            batch = [transform(sample).to_tensor() for sample in batch]
            batch = batch[0].batched(batch)
        return batch

    @abstractmethod
    def training_step(
        self, batch: BaseDataInstance, training_engine: Engine | None = None, **kwargs
    ) -> ModelOutput:
        """
        Defines the training step for the model.

        Args:
            batch ("BaseDataInstance"): The input batch.
            training_engine (Optional[Engine]): The training engine instance.
            **kwargs: Additional arguments.

        Returns:
            AtriaModelOutput: The output of the model.
        """

    @abstractmethod
    def evaluation_step(
        self,
        batch: BaseDataInstance,
        evaluation_engine: Engine | None = None,
        training_engine: Engine | None = None,
        stage: TrainingStage = TrainingStage.test,
        **kwargs,
    ) -> ModelOutput:
        """
        Defines the evaluation step for the model.

        Args:
            batch ("BaseDataInstance"): The input batch.
            evaluation_engine (Optional[Engine]): The evaluation engine instance.
            training_engine (Optional[Engine]): The training engine instance.
            stage (TrainingStage): The current training stage.
            **kwargs: Additional arguments.

        Returns:
            AtriaModelOutput: The output of the model.
        """

    @abstractmethod
    def predict_step(
        self, batch: BaseDataInstance, evaluation_engine: Engine | None = None, **kwargs
    ) -> ModelOutput:
        """
        Defines the prediction step for the model.

        Args:
            batch ("BaseDataInstance"): The input batch.
            evaluation_engine (Optional[Engine]): The evaluation engine instance.
            **kwargs: Additional arguments.

        Returns:
            AtriaModelOutput: The output of the model.
        """

    def visualization_step(
        self,
        batch: BaseDataInstance,
        evaluation_engine: Engine | None = None,
        training_engine: Engine | None = None,
        **kwargs,
    ) -> None:
        """
        Defines the visualization step for the model.

        Args:
            batch ("BaseDataInstance"): The input batch.
            evaluation_engine (Optional[Engine]): The evaluation engine instance.
            training_engine (Optional[Engine]): The training engine instance.
            **kwargs: Additional arguments.
        """
        raise NotImplementedError(
            "Visualization step is not implemented for this model pipeline."
        )

    def __repr__(self):
        """
        Returns a string representation of the model pipeline.

        Returns:
            str: The string representation.
        """
        from atria_models.utilities.nn_modules import _summarize_model

        return f"{self.__class__.__name__}:\n{_summarize_model(self)}"

    def __str__(self):
        """
        Returns a string representation of the model pipeline.

        Returns:
            str: The string representation.
        """
        return self.__repr__()

    def _validate_model_config(self) -> None:
        """
        Validates the input model.

        Args:
            model (Union[AtriaModel, AtriaModelConfigDict]): The input model.

        Raises:
            AssertionError: If the model does not meet the required criteria.
        """
        if self.__requires_model_dict__:
            assert isinstance(self.config.model, dict), (
                f"Model builder must be provided as a dictionary of "
                f"{AtriaModel} when _REQUIRES_MODEL_DICT is True"
            )
            for key in self.__required_model_dict_keys__:
                assert key in self.config.model, (
                    f"Input model dictionary {self.config.model} must contain the key {key}"
                )
        else:
            assert isinstance(self.config.model, AtriaModel), (
                f"Expected model type to be {AtriaModel}, but got {type(self.config.model)}"
            )

    def _build_metrics(self, device) -> dict[str, Callable]:
        """
        Builds metrics for the model using the provided metric factory.

        Raises:
            ValueError: If the metric factory is not a dictionary.
        """

        metrics = {}
        for key, metric_builder in self.config.metric_builders.items():
            kwargs = {}
            metric_builder.get_possible_args()
            if "num_classes" in metric_builder.get_possible_args():
                assert self._dataset_metadata is not None, (
                    "Dataset metadata must be provided to determine the number of classes."
                )
                assert (
                    self._dataset_metadata.dataset_labels.classification is not None
                ), (
                    f"`instance_classification` dataset labels must be provided for {self.__class__.__name__} "
                    "to build classification metrics."
                )
                kwargs["num_classes"] = len(
                    self._dataset_metadata.dataset_labels.classification
                )
                kwargs["device"] = device
            metrics[key] = metric_builder(**kwargs)
        return metrics

    def _build_model(self) -> AtriaModel | AtriaModelDict:
        """
        Builds the model using the provided model builders.

        Returns:
            Union[AtriaModel, AtriaModelDict]: The built model.
        """
        from torch import nn

        model_kwargs = self._prepare_build_kwargs()
        if isinstance(self.config.model, dict):
            assert isinstance(model_kwargs, dict), (
                "Model kwargs must be provided as a dictionary when model_factory is a dictionary"
            )
            assert sorted(model_kwargs.keys()) == sorted(self.config.model.keys()), (
                f"Model kwargs must be a dictionary with the same keys as the model builders. "
                f"Got {model_kwargs.keys()} and {self.config.model.keys()}"
            )

            trainable_models = {}
            non_trainable_models = {}
            for key, m in self.config.model.items():
                m = m.build(**model_kwargs[key])
                if not m.is_frozen:
                    trainable_models[key] = m
                else:
                    non_trainable_models[key] = m

            return AtriaModelDict(
                trainable_models=nn.ModuleDict(trainable_models),
                non_trainable_models=nn.ModuleDict(non_trainable_models),
            )
        else:
            return self.config.model.build(**model_kwargs)

    def _prepare_build_kwargs(self) -> dict[str, dict[str, Any]] | dict[str, Any]:
        """
        Prepares the keyword arguments for building the model.

        Returns:
            Union[Dict[str, Dict[str, Any]], Dict[str, Any]]: The prepared keyword arguments.
        """
        if isinstance(self._model, dict):
            return {key: {} for key in self._model}
        else:
            return {}
