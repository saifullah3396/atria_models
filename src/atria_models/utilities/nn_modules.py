"""
Neural Network Modules Utilities

This module provides utility functions for managing and manipulating PyTorch neural network modules.
These utilities include functions for setting and replacing modules, freezing layers, summarizing models,
and converting batch normalization layers to group normalization.

Functions:
    - _rsetattr: Recursively sets an attribute on an object.
    - _rgetattr: Recursively gets an attribute from an object.
    - _set_module_with_name: Sets a module in a model by name.
    - _replace_module_with_name: Replaces a module in a model by name.
    - _get_all_nn_modules_in_object: Retrieves all `torch.nn.Module` instances in an object.
    - _get_last_module: Retrieves the last module in a model.
    - _find_layer_in_model: Finds a specific layer in a model by name.
    - _freeze_layers_by_name: Freezes specific layers in a model by name.
    - _freeze_layers: Freezes a list of layers in a model.
    - _summarize_model: Summarizes the components of a model.
    - _batch_norm_to_group_norm: Converts batch normalization layers to group normalization.
    - _remove_lora_layers: Removes LoRA layers from a model.
    - _get_logits_from_output: Extracts logits from a model's output.
    - _convert_label_tensors_to_tags: Converts label tensors to human-readable tags.
    - _validate_built_model: Validates the structure of a built model.
    - _unnormalize_image: Unnormalizes an image tensor using mean and standard deviation.

Dependencies:
    - torch: For PyTorch operations.
    - functools: For recursive attribute manipulation.
    - dataclasses: For checking dataclass objects.
    - atria_core.logger: For logging utilities.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

import functools
from dataclasses import is_dataclass
from typing import TYPE_CHECKING, Any, Union

from atria_core.logger import get_logger
from pydantic import BaseModel, field_validator

if TYPE_CHECKING:
    import torch
    from torch import nn

logger = get_logger(__name__)


class AtriaModelDict(BaseModel):
    """
    A data model for managing trainable and non-trainable PyTorch models.

    Attributes:
        trainable_models (nn.ModuleDict): A dictionary of trainable models.
        non_trainable_models (nn.ModuleDict): A dictionary of non-trainable models.
    """

    trainable_models: Any
    non_trainable_models: Any

    @field_validator("trainable_models", "non_trainable_models")
    @classmethod
    def check_module_dict(cls, v: "nn.ModuleDict") -> "nn.ModuleDict":
        """
        Validates that the provided value is an nn.ModuleDict.

        Args:
            v (nn.ModuleDict): The value to validate.

        Returns:
            nn.ModuleDict: The validated value.

        Raises:
            ValueError: If the value is not an nn.ModuleDict.
        """
        from torch import nn

        if not isinstance(v, nn.ModuleDict):
            raise ValueError("Must be a nn.ModuleDict")
        return v


def _rsetattr(obj: Any, attr: str, val: Any) -> None:
    """
    Recursively sets an attribute on an object.

    Args:
        obj (Any): The object to set the attribute on.
        attr (str): The attribute name, which can include nested attributes separated by dots.
        val (Any): The value to set for the attribute.
    """
    pre, _, post = attr.rpartition(".")
    return setattr(_rgetattr(obj, pre) if pre else obj, post, val)


def _rgetattr(obj: Any, attr: str, *args: Any) -> Any:
    """
    Recursively gets an attribute from an object.

    Args:
        obj (Any): The object to get the attribute from.
        attr (str): The attribute name, which can include nested attributes separated by dots.
        *args (Any): Default value to return if the attribute is not found.

    Returns:
        Any: The value of the attribute.
    """

    def _getattr(obj: Any, attr: str) -> Any:
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def _set_module_with_name(
    model: "nn.Module", module_name: str, module: "nn.Module"
) -> None:
    """
    Sets a module in a model by name.

    Args:
        model (nn.Module): The model to modify.
        module_name (str): The name of the module to set.
        module (nn.Module): The module to set.
    """
    return setattr(model, module_name, module)


def _replace_module_with_name(
    module: "nn.Module", target_name: str, new_module: "nn.Module"
) -> None:
    """
    Replaces a module in a model by name.

    Args:
        module (nn.Module): The parent module containing the target module.
        target_name (str): The name of the module to replace, which can include nested names separated by dots.
        new_module (nn.Module): The new module to replace the target module with.
    """
    target_name = target_name.split(".")
    if len(target_name) > 1:
        _replace_module_with_name(
            getattr(module, target_name[0]), ".".join(target_name[1:]), new_module
        )
    setattr(module, target_name[-1], new_module)


def _get_all_nn_modules_in_object(object: "nn.Module") -> dict[str, "nn.Module"]:
    """
    Retrieves all `nn.Module` instances in an object.

    Args:
        object (nn.Module): The object to search for modules.

    Returns:
        Dict[str, nn.Module]: A dictionary mapping module names to module instances.
    """
    from torch import nn

    return {k: v for k, v in object.__dict__.items() if isinstance(v, nn.Module)}


def _get_last_module(model: "nn.Module") -> Any:
    """
    Retrieves the last module in a model.

    Args:
        model (nn.Module): The model to search.

    Returns:
        Any: The last module in the model.
    """
    return list(model.named_modules())[-1]


def _find_layer_in_model(model: "nn.Module", layer_name: str) -> str:
    """
    Finds a specific layer in a model by name.

    Args:
        model (nn.Module): The model to search.
        layer_name (str): The name of the layer to find.

    Returns:
        str: The name of the layer.

    Raises:
        ValueError: If the layer is not found in the model.
    """
    layer = [x for x, m in model.named_modules() if x == layer_name]
    if len(layer) == 0:
        raise ValueError(f"Encoder layer {layer_name} not found in the model.")
    return layer[0]


def _freeze_layers_by_name(model: "nn.Module", layer_names: list[str]) -> None:
    """
    Freezes specific layers in a model by name.

    Args:
        model (nn.Module): The model to modify.
        layer_names (List[str]): A list of layer names to freeze.
    """
    for layer_name in layer_names:
        layer = _find_layer_in_model(model, layer_name)
        for p in layer.parameters():
            p.requires_grad = False


def _freeze_layers(layers: list["nn.Module"]) -> None:
    """
    Freezes a list of layers in a model.

    Args:
        layers (List[nn.Module]): A list of layers to freeze.
    """
    for layer in layers:
        for p in layer.parameters():
            p.requires_grad = False


def _summarize_model(model: object) -> None:
    """
    Summarizes the components of a model.

    Args:
        model (object): The model to summarize.

    Returns:
        str: A string representation of the model summary.
    """
    from torch import nn
    from torchinfo import summary

    nn_module_dict = nn.ModuleDict()
    if isinstance(model._model, AtriaModelDict):
        nn_module_dict.add_module("trainable_models", model._model.trainable_models)
        nn_module_dict.add_module(
            "non_trainable_models", model._model.non_trainable_models
        )
    for k, v in _get_all_nn_modules_in_object(model).items():
        nn_module_dict.add_module(k, v)
    return str(summary(nn_module_dict, verbose=0, depth=3))


def _batch_norm_to_group_norm(model: "nn.Module") -> None:
    """
    Converts batch normalization layers to group normalization.

    Args:
        model (nn.Module): The model to modify.
    """
    import torch

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            num_channels = module.num_features

            def get_groups(num_channels: int, groups: int) -> int:
                """
                Recursively determines the number of groups for group normalization.

                Args:
                    num_channels (int): The number of channels.
                    groups (int): The initial number of groups.

                Returns:
                    int: The adjusted number of groups.
                """
                if num_channels % groups != 0:
                    groups = groups // 2
                    groups = get_groups(num_channels, groups)
                return groups

            groups = get_groups(num_channels, 32)
            gn = torch.nn.GroupNorm(groups, num_channels)
            _rsetattr(model, name, gn)


def _freeze_layers_with_key_pattern(
    self,
    model: "nn.Module",
    frozen_keys_patterns: list[str],
    unfrozen_keys_patterns: list[str],
) -> None:
    """
    Apply layer-specific freezing/unfreezing based on name patterns.

    Args:
        model (nn.Module): The model to modify.
        frozen_keys_patterns (List[str]): Patterns for layers to freeze.
        unfrozen_keys_patterns (List[str]): Patterns for layers to unfreeze.

    Returns:
        Dict[str, bool]: A dictionary of parameter names and their `requires_grad` status.
    """
    for name, param in model.named_parameters():
        if any(pattern in name for pattern in frozen_keys_patterns):
            param.requires_grad = False
        if any(pattern in name for pattern in unfrozen_keys_patterns):
            param.requires_grad = True
    trainable_params = {
        name: param.requires_grad for name, param in model.named_parameters()
    }
    return trainable_params


def _remove_lora_layers(model: "nn.Module") -> None:
    """
    Removes LoRA layers from a model.

    Args:
        model (nn.Module): The model to modify.
    """
    from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear
    from torch import nn

    for name, module in model.named_modules():
        if isinstance(module, LoRACompatibleLinear):
            new_module = nn.Linear(module.in_features, module.out_features)
            new_module.__dict__.update(module.__dict__)
            _rsetattr(model, name, new_module)
        if isinstance(module, LoRACompatibleConv):
            new_module = nn.Conv2d(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                module.stride,
                module.padding,
                module.dilation,
                module.groups,
            )
            _rsetattr(model, name, new_module)


def _get_logits_from_output(model_output: Any) -> "torch.Tensor":
    """
    Extracts logits from a model's output.

    Args:
        model_output (Any): The output of the model.

    Returns:
        torch.Tensor: The logits extracted from the model output.

    Raises:
        ValueError: If logits cannot be extracted from the model output.
    """
    import torch

    if isinstance(model_output, torch.Tensor):
        return model_output
    elif isinstance(model_output, tuple):
        return model_output[0]
    elif is_dataclass(model_output):
        return model_output.logits
    elif isinstance(model_output, dict):
        if "logits" in model_output:
            return model_output["logits"]
    else:
        raise ValueError(f"Could not extract logits from model output: {model_output}")


def _convert_label_tensors_to_tags(
    class_labels: list[str],
    predicted_labels: "torch.Tensor",
    target_labels: "torch.Tensor",
    ignore_label: int = -100,
):
    """
    Converts label tensors to human-readable tags.

    Args:
        class_labels (List[str]): The list of class labels.
        predicted_labels (torch.Tensor): The predicted labels as a tensor.
        target_labels (torch.Tensor): The target labels as a tensor.
        ignore_label (int): The label to ignore during conversion. Defaults to -100.

    Returns:
        Tuple[List[List[str]], List[List[str]]]: The predicted and target labels as human-readable tags.
    """
    predicted_labels = predicted_labels.detach().cpu().tolist()
    target_labels = target_labels.detach().cpu().tolist()
    predicted_labels = [
        [
            class_labels[pred_label]
            for (pred_label, target_label) in zip(prediction, target, strict=True)
            if target_label != ignore_label
        ]
        for prediction, target in zip(predicted_labels, target_labels, strict=True)
    ]
    target_labels = [
        [
            class_labels[target_label]
            for target_label in target
            if target_label != ignore_label
        ]
        for target in target_labels
    ]
    return predicted_labels, target_labels


def _validate_built_model(torch_model: Any) -> None:
    """
    Validates the structure of a built model.

    Args:
        torch_model (Any): The model to validate.

    Returns:
        Any: The validated model.

    Raises:
        AssertionError: If the model structure is invalid.
    """
    from torch import nn

    if isinstance(torch_model, AtriaModelDict):
        for key, model in torch_model.trainable_models.items():
            assert isinstance(model, nn.Module), (
                f"Model must be a torch nn.Module. Got {type(model)} for key {key}"
            )
        for key, model in torch_model.non_trainable_models.items():
            assert isinstance(model, nn.Module), (
                f"Model must be a torch nn.Module. Got {type(model)} for key {key}"
            )
    else:
        assert isinstance(torch_model, nn.Module), (
            f"Model must be a torch nn.Module. Got {type(torch_model)}"
        )
    return torch_model


def _unnormalize_image(
    image: "torch.Tensor", mean: "torch.Tensor", std: "torch.Tensor"
) -> "torch.Tensor":
    """
    Unnormalizes an image tensor using the given mean and standard deviation.

    Args:
        image (torch.Tensor): The image tensor to unnormalize.
        mean (torch.Tensor): The mean values for each channel.
        std (torch.Tensor): The standard deviation values for each channel.

    Returns:
        torch.Tensor: The unnormalized image tensor.
    """
    from torchvision.transforms.functional import normalize

    return normalize(image=image, mean=-mean, std=1 / std, inplace=False)


def _module_to_device(
    module: "nn.Module",
    device: Union[str, "torch.device"],
    sync_bn: bool = False,
    prepare_for_distributed: bool = False,
) -> "nn.Module":
    """
    Moves a PyTorch module to a specified device and optionally prepares it for distributed training.

    Args:
        module (nn.Module): The PyTorch module to move.
        device (Union[str, torch.device]): The target device (e.g., 'cpu', 'cuda').
        sync_bn (bool, optional): Whether to synchronize batch normalization across devices. Defaults to False.
        prepare_for_distributed (bool, optional): Whether to prepare the module for distributed training. Defaults to False.

    Returns:
        nn.Module: The module moved to the specified device and optionally prepared for distributed training.

    Notes:
        - If `prepare_for_distributed` is True, the module is wrapped for distributed training using Ignite's `auto_model`.
        - If the module is already wrapped in `DistributedDataParallel` or `DataParallel`, it is further wrapped with `ModuleProxyWrapper`.
    """
    import ignite.distributed as idist
    import torch
    from torch import nn

    from atria_models.utilities.ddp_model_proxy import ModuleProxyWrapper

    if prepare_for_distributed:
        module = idist.auto_model(
            module, sync_bn=(False if device == torch.device("cpu") else sync_bn)
        )
    else:
        module.to(device)

    if isinstance(module, nn.parallel.DistributedDataParallel | nn.DataParallel):
        module = ModuleProxyWrapper(module)

    return module
