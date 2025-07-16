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

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Optional, TypeVar, Union

from atria_core.logger import get_logger
from atria_core.transforms.base import DataTransform
from atria_core.types import BaseDataInstance, TrainingStage

from atria_models.core.atria_model import AtriaModel
from atria_models.utilities.checkpoints import CheckpointConfig
from atria_models.utilities.config import setup_model_pipeline_config

if TYPE_CHECKING:
    import torch
    from atria_core.types import DatasetMetadata
    from ignite.contrib.handlers import TensorboardLogger
    from ignite.engine import Engine
    from ignite.handlers import ProgressBar

    from atria_models.outputs import ModelOutput
    from atria_models.utilities.nn_modules import AtriaModelDict

logger = get_logger(__name__)

BaseDataInstanceType = TypeVar("BaseDataInstance", bound=BaseDataInstance)


@setup_model_pipeline_config()
class AtriaModelPipeline(ABC):
    """
    Abstract base class for task-specific PyTorch models.

    This class provides functionality for building models, managing checkpoints,
    handling distributed training, and defining task-specific forward passes.

    Attributes:
        _REQUIRES_MODEL_DICT (bool): Whether the model builder must be a dictionary.
        _REQUIRED_MODEL_KEYS (List[str]): Required keys for the model builder dictionary.
    """

    _REQUIRES_MODEL_DICT: ClassVar[bool] = False
    _REQUIRED_MODEL_KEYS: ClassVar[list[str]] = []

    def __init__(
        self,
        model: AtriaModel | dict[str, AtriaModel],
        checkpoint_configs: list[CheckpointConfig] | None = None,
        input_transform: DataTransform | None = None,
    ):
        """
        Initializes the AtriaModelPipeline instance.

        Args:
            model (Union[AtriaModel, Dict[str, AtriaModel]]): The model or dictionary of models.
            checkpoint_configs (Optional[List[CheckpointConfig]]): List of checkpoint configurations.
        """
        self._model = model
        self._checkpoint_configs = checkpoint_configs
        self._input_transform = input_transform
        self._progress_bar = None
        self._validate_model_input(model=model)

    @property
    def model_name(self) -> str:
        """
        Returns the name of the model.

        Returns:
            str: The model name.
        """
        return self._model.model_name

    @property
    def task_module_name(self) -> str:
        """
        Returns the name of the task module.

        Returns:
            str: The task module name.
        """
        return self.__class__.__name__

    @property
    def model(self) -> Union["torch.nn.Module", "AtriaModelDict"]:
        """
        Returns the underlying PyTorch model.

        Returns:
            Union[torch.nn.Module, AtriaModelDict]: The PyTorch model.
        """
        return self._model

    @property
    def ema_modules(self) -> Union["torch.nn.Module", dict[str, "torch.nn.Module"]]:
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
    def progress_bar(self) -> "ProgressBar":
        """
        Returns the progress bar for the model.

        Returns:
            ProgressBar: The progress bar instance.
        """
        return self._progress_bar

    @progress_bar.setter
    def progress_bar(self, progress_bar: "ProgressBar") -> None:
        """
        Sets the progress bar for the model.

        Args:
            progress_bar (ProgressBar): The progress bar instance.
        """
        self._progress_bar = progress_bar

    def build(
        self,
        dataset_metadata: Optional["DatasetMetadata"] = None,
        tb_logger: Optional["TensorboardLogger"] = None,
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

        if self._checkpoint_configs is not None:
            CheckpointManager.load_checkpoints(
                model=self._model, checkpoint_configs=self._checkpoint_configs
            )

        if idist.get_rank() == 0:
            idist.barrier()

        return self

    def to_device(self, device: "torch.device", sync_bn: bool = False) -> None:
        """
        Moves the model to the specified device.

        Args:
            device (torch.device): The target device.
            sync_bn (bool): Whether to synchronize BatchNorm layers across devices. Defaults to False.
        """
        from atria_models.utilities.nn_modules import AtriaModelDict, _module_to_device

        if isinstance(self._model, AtriaModelDict):
            logger.info(f"Moving trainable_models to {device}")
            self._model.trainable_models = _module_to_device(
                self._model.trainable_models,
                device=device,
                sync_bn=sync_bn,
                prepare_for_distributed=True,
            )
            logger.info(f"Moving non_trainable_models to {device}")
            self._model.non_trainable_models = _module_to_device(
                self._model.non_trainable_models,
                device=device,
                sync_bn=sync_bn,
                prepare_for_distributed=False,
            )
        else:
            logger.info(f"Moving model to {device}")
            self._model = _module_to_device(
                self._model,
                device=device,
                sync_bn=sync_bn,
                prepare_for_distributed=True,
            )

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

    def optimizer_parameters(self) -> Mapping[str, list["torch.nn.Parameter"]]:
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

    @abstractmethod
    def training_step(
        self,
        batch: BaseDataInstanceType,
        training_engine: Optional["Engine"] = None,
        **kwargs,
    ) -> "ModelOutput":
        """
        Defines the training step for the model.

        Args:
            batch (BaseDataInstanceType): The input batch.
            training_engine (Optional[Engine]): The training engine instance.
            **kwargs: Additional arguments.

        Returns:
            AtriaModelOutput: The output of the model.
        """

    @abstractmethod
    def evaluation_step(
        self,
        batch: BaseDataInstanceType,
        evaluation_engine: Optional["Engine"] = None,
        training_engine: Optional["Engine"] = None,
        stage: TrainingStage = TrainingStage.test,
        **kwargs,
    ) -> "ModelOutput":
        """
        Defines the evaluation step for the model.

        Args:
            batch (BaseDataInstanceType): The input batch.
            evaluation_engine (Optional[Engine]): The evaluation engine instance.
            training_engine (Optional[Engine]): The training engine instance.
            stage (TrainingStage): The current training stage.
            **kwargs: Additional arguments.

        Returns:
            AtriaModelOutput: The output of the model.
        """

    @abstractmethod
    def predict_step(
        self,
        batch: BaseDataInstanceType,
        evaluation_engine: Optional["Engine"] = None,
        **kwargs,
    ) -> "ModelOutput":
        """
        Defines the prediction step for the model.

        Args:
            batch (BaseDataInstanceType): The input batch.
            evaluation_engine (Optional[Engine]): The evaluation engine instance.
            **kwargs: Additional arguments.

        Returns:
            AtriaModelOutput: The output of the model.
        """

    def visualization_step(
        self,
        batch: BaseDataInstanceType,
        evaluation_engine: Optional["Engine"] = None,
        training_engine: Optional["Engine"] = None,
        **kwargs,
    ) -> None:
        """
        Defines the visualization step for the model.

        Args:
            batch (BaseDataInstanceType): The input batch.
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

    def _validate_model_input(
        self, model: Union["AtriaModel", "AtriaModelDict"]
    ) -> None:
        """
        Validates the input model.

        Args:
            model (Union[AtriaModel, AtriaModelDict]): The input model.

        Raises:
            AssertionError: If the model does not meet the required criteria.
        """
        if self._REQUIRES_MODEL_DICT:
            assert isinstance(model, dict), (
                f"Model builder must be provided as a dictionary of "
                f"{AtriaModel} when _REQUIRES_MODEL_DICT is True"
            )
            for key in self._REQUIRED_MODEL_KEYS:
                assert key in model, (
                    f"Input model dictionary {model} must contain the key {key}"
                )
        else:
            assert isinstance(model, AtriaModel), (
                f"Expected model type to be {AtriaModel}, but got {type(model)}"
            )

    def _build_model(self) -> Union["AtriaModel", "AtriaModelDict"]:
        """
        Builds the model using the provided model builders.

        Returns:
            Union[AtriaModel, AtriaModelDict]: The built model.
        """
        from torch import nn

        model_kwargs = self._prepare_build_kwargs()
        if isinstance(self._model, dict):
            assert isinstance(model_kwargs, dict), (
                "Model kwargs must be provided as a dictionary when model_factory is a dictionary"
            )
            assert sorted(model_kwargs.keys()) == sorted(self._model.keys()), (
                f"Model kwargs must be a dictionary with the same keys as the model builders. "
                f"Got {model_kwargs.keys()} and {self._model.keys()}"
            )

            trainable_models = {}
            non_trainable_models = {}
            for key, m in self._model.items():
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
            return self._model.build(**model_kwargs)

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
