"""
Checkpoint Utilities Module

This module provides utility functions for managing and manipulating model checkpoints.
These utilities include loading checkpoints, filtering and modifying checkpoint keys, and
loading checkpoints into PyTorch models.

Functions:
    - _get_filesystem: Retrieves the filesystem for a given path or URL.
    - _filter_keys: Removes specific keys from checkpoint state dictionaries.
    - _prepend_keys: Prepends specific keys to checkpoint state dictionaries.
    - _replace_keys: Replaces specific keys in checkpoint state dictionaries.
    - _filter_with_prefix: Filters checkpoint keys based on a prefix.
    - _load_checkpoint: Loads a checkpoint from a file or URL.
    - _load_checkpoint_into_model: Loads a checkpoint into a PyTorch model.

Dependencies:
    - torch: For PyTorch operations.
    - fsspec: For handling file systems and URLs.
    - atria_core.logger: For logging utilities.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from atria_core.logger import get_logger
from fsspec.core import url_to_fs
from fsspec.implementations.local import AbstractFileSystem
from pydantic import BaseModel

if TYPE_CHECKING:
    import torch
    from torch import nn


logger = get_logger(__name__)


class CheckpointConfig(BaseModel):
    """
    CheckpointConfig is a configuration model for managing checkpoint-related settings.

    Attributes:
        checkpoint_path (str): The path to the main checkpoint file.
        checkpoint_state_dict_path (Optional[str]): The path to the checkpoint's state dictionary file. Defaults to None.
        model_state_dict_path (Optional[str]): The path to the model's state dictionary file. Defaults to None.
        load_checkpoint_strict (bool): A flag indicating whether to strictly enforce that the keys in the checkpoint match
            the keys in the model. Defaults to False.
    """

    checkpoint_path: str
    checkpoint_state_dict_path: str | None = None
    model_state_dict_path: str | None = None
    load_checkpoint_strict: bool = False


class CheckpointManager:
    """
    CheckpointManager provides static methods for loading and applying checkpoints to PyTorch models.

    Methods:
        - load_checkpoints: Loads multiple checkpoints into a model based on a list of configurations.
        - _resolve_state_dict_path: Resolves a nested path within a checkpoint's state dictionary.
        - _load_checkpoint_from_path_or_url: Loads a checkpoint from a file or URL.
        - _apply_checkpoint_to_model: Applies a checkpoint to a PyTorch model.
    """

    @staticmethod
    def load_checkpoints(
        model: "torch.nn.Module", checkpoint_configs: list[CheckpointConfig]
    ) -> None:
        """
        Loads multiple checkpoints into a model based on a list of configurations.

        Args:
            model (torch.nn.Module): The PyTorch model to load the checkpoints into.
            checkpoint_configs (List[CheckpointConfig]): A list of checkpoint configurations.

        Returns:
            None
        """
        for checkpoint_config in checkpoint_configs:
            if not checkpoint_config.checkpoint_path:
                logger.info("No checkpoint path provided. Skipping checkpoint loading.")
                continue

            logger.info(
                f"Loading checkpoint from path: {checkpoint_config.checkpoint_path}"
            )
            checkpoint = CheckpointManager._load_checkpoint_from_path_or_url(
                checkpoint_config.checkpoint_path
            )
            if checkpoint_config.checkpoint_state_dict_path:
                checkpoint = CheckpointManager._resolve_state_dict_path(
                    checkpoint,
                    checkpoint_config.checkpoint_state_dict_path,
                    checkpoint_config.model_state_dict_path,
                )
            logger.info(
                f"Loading checkpoint into model at path [{checkpoint_config.model_state_dict_path}] "
                f"(state dict path: [{checkpoint_config.checkpoint_state_dict_path}], strict={checkpoint_config.load_checkpoint_strict})"
            )
            CheckpointManager._apply_checkpoint_to_model(
                model=model,
                checkpoint=checkpoint,
                strict=checkpoint_config.load_checkpoint_strict,
            )

    @staticmethod
    def load_checkpoint(
        model: "torch.nn.Module",
        checkpoint: dict[str, Any],
        checkpoint_state_dict_path: str | None = "state_dict",
        model_state_dict_path: str | None = None,
        load_checkpoint_strict: bool = False,
    ) -> None:
        checkpoint = CheckpointManager._resolve_state_dict_path(
            checkpoint, checkpoint_state_dict_path, model_state_dict_path
        )
        logger.info(
            f"Loading checkpoint into model at path [{model_state_dict_path}] "
            f"(state dict path: [{checkpoint_state_dict_path}], strict={load_checkpoint_strict})"
        )
        CheckpointManager._apply_checkpoint_to_model(
            model=model, checkpoint=checkpoint, strict=load_checkpoint_strict
        )
        return checkpoint

    @staticmethod
    def _resolve_state_dict_path(
        checkpoint: dict[str, Any],
        checkpoint_path_str: str,
        model_state_dict_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Resolves a nested path within a checkpoint's state dictionary.

        Args:
            checkpoint (Dict[str, Any]): The checkpoint state dictionary.
            path_str (str): The dot-separated path string to resolve.

        Returns:
            Dict[str, Any]: The resolved state dictionary.

        Raises:
            KeyError: If the path part is not found in the checkpoint.
        """
        resolved = checkpoint

        if checkpoint_path_str is not None and checkpoint_path_str != "":
            path_parts = checkpoint_path_str.split(".")
            for part in path_parts:
                if part in resolved:
                    resolved = resolved[part]
                else:
                    # Try to match key by prefix
                    matching_keys = [k for k in resolved if part in k]
                    if matching_keys:
                        resolved = _filter_with_prefix(resolved, part)
                    else:
                        available_keys = ", ".join(list(resolved.keys())[:10])
                        raise KeyError(
                            f"Path part '{part}' not found. Available keys: {available_keys}..."
                        )

        if model_state_dict_path is not None:
            parts = model_state_dict_path.split(".")
            resolved_nested = resolved
            for key in reversed(parts):
                resolved_nested = {key: resolved_nested}
            return resolved_nested

        return resolved

    @staticmethod
    def _load_checkpoint_from_path_or_url(path_or_url: str | Path) -> Any:
        """
        Loads a checkpoint from a file or URL.

        Args:
            path_or_url (Union[str, Path]): The path or URL of the checkpoint.

        Returns:
            Any: The loaded checkpoint.

        Raises:
            Exception: If an error occurs while loading the checkpoint.
        """
        import ignite.distributed as idist
        import torch

        map_location = idist.device()
        if idist.get_world_size() > 1:
            map_location = "cpu"

        if str(path_or_url).startswith("http"):
            return torch.hub.load_state_dict_from_url(
                str(path_or_url), map_location=map_location
            )
        if str(path_or_url).startswith("hf://"):
            from transformers import (
                AutoModel,
                AutoModelForQuestionAnswering,
                AutoModelForSequenceClassification,
                AutoModelForTokenClassification,
            )

            HF_MODEL_MAP = OrderedDict(
                {
                    "hf://sequence_classification/": AutoModelForSequenceClassification,
                    "hf://token_classification/": AutoModelForTokenClassification,
                    "hf://question_answering/": AutoModelForQuestionAnswering,
                    "hf://": AutoModel,
                }
            )

            for key in HF_MODEL_MAP:
                if str(path_or_url).startswith(key):
                    model_class = HF_MODEL_MAP[key]
                    path_or_url = str(path_or_url).replace(key, "")
                    model = model_class.from_pretrained(path_or_url)
                    return model.state_dict()
        fs = _get_filesystem(path_or_url)
        try:
            with fs.open(path_or_url, "rb") as f:
                return torch.load(f, map_location=map_location)
        except Exception as e:
            logger.error(f"Error loading the checkpoint: {e}")
            raise e

    @staticmethod
    def _apply_checkpoint_to_model(
        model: "nn.Module",
        checkpoint: dict[str, Any],
        model_state_dict_path: str | None = None,
        strict: bool = True,
    ) -> None:
        """
        Applies a checkpoint to a PyTorch model.

        Args:
            model (nn.Module): The model to load the checkpoint into.
            checkpoint (Dict[str, Any]): The checkpoint state dictionary.
            model_state_dict_path (Optional[str]): The path to the model's state dictionary.
            strict (bool): Whether to enforce strict loading of the checkpoint. Defaults to True.

        Raises:
            RuntimeError: If the model is a `TorchModelDict` and cannot load the checkpoint.
        """
        from ignite.handlers import Checkpoint

        Checkpoint.load_objects(
            to_load={key: getattr(model, key) for key in checkpoint.keys()},
            checkpoint=checkpoint,
            strict=strict,
        )


def _get_filesystem(path: Path, **kwargs: Any) -> AbstractFileSystem:
    """
    Retrieves the filesystem for a given path or URL.

    Args:
        path (Path): The path or URL to retrieve the filesystem for.
        **kwargs (Any): Additional arguments for the filesystem.

    Returns:
        AbstractFileSystem: The filesystem object.
    """
    fs, _ = url_to_fs(str(path), **kwargs)
    return fs


def _filter_keys(checkpoint: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    """
    Removes specific keys from checkpoint state dictionaries.

    Args:
        checkpoint (Dict[str, Any]): The checkpoint state dictionary.
        keys (List[str]): The keys to remove.

    Returns:
        Dict[str, Any]: The filtered checkpoint state dictionary.
    """
    checkpoint_filtered = {}
    for state in checkpoint:
        updated_state = state
        for key in keys:
            if key in updated_state:
                updated_state = updated_state.replace(key, "")
        checkpoint_filtered[updated_state] = checkpoint[state]
    return checkpoint_filtered


def _prepend_keys(checkpoint: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    """
    Prepends specific keys to checkpoint state dictionaries.

    Args:
        checkpoint (Dict[str, Any]): The checkpoint state dictionary.
        keys (List[str]): The keys to prepend.

    Returns:
        Dict[str, Any]: The updated checkpoint state dictionary.
    """
    checkpoint_prepended = {}
    for state in checkpoint:
        updated_state = state
        for key in keys:
            if key not in updated_state:
                updated_state = key + updated_state
        checkpoint_prepended[updated_state] = checkpoint[state]
    return checkpoint_prepended


def _replace_keys(
    checkpoint: dict[str, Any], key: str, replacement: str
) -> dict[str, Any]:
    """
    Replaces specific keys in checkpoint state dictionaries.

    Args:
        checkpoint (Dict[str, Any]): The checkpoint state dictionary.
        key (str): The key to replace.
        replacement (str): The replacement value.

    Returns:
        Dict[str, Any]: The updated checkpoint state dictionary.
    """
    checkpoint_filtered = {}
    for state in checkpoint:
        updated_state = state
        if key in updated_state:
            updated_state = updated_state.replace(key, replacement)
        checkpoint_filtered[updated_state] = checkpoint[state]
    return checkpoint_filtered


def _filter_with_prefix(checkpoint: dict[str, Any], prefix_key: str) -> dict[str, Any]:
    """
    Filters checkpoint keys based on a prefix.

    Args:
        checkpoint (Dict[str, Any]): The checkpoint state dictionary.
        prefix_key (str): The prefix to filter keys by.

    Returns:
        Dict[str, Any]: The filtered checkpoint state dictionary.
    """
    checkpoint_filtered = {}
    for state in checkpoint:
        if state.startswith(prefix_key):
            checkpoint_filtered[state[len(prefix_key) + 1 :]] = checkpoint[state]
    return checkpoint_filtered
