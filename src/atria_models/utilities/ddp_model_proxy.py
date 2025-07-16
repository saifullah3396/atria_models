"""
Module Proxy Wrapper

This module defines the `ModuleProxyWrapper` class, which is used to wrap a `DistributedDataParallel` (DDP) module.
The wrapper forwards requests for missing attributes to the module wrapped by DDP (the twice-wrapped module).
It also forwards calls to `state_dict` and `load_state_dict` methods, ensuring compatibility with PyTorch's DDP.

Classes:
    - ModuleProxyWrapper: A wrapper for DDP modules that forwards attribute and method calls to the underlying module.

Usage Example:
    wrapped_module = DistributedDataParallel(module, **ddp_args)
    wrapped_module = ModuleProxyWrapper(wrapped_module)
    assert wrapped_module.state_dict().keys() == module.state_dict().keys()

Dependencies:
    - torch.nn: For PyTorch's neural network module.

Author: Facebook
Date: 2025-04-14
Version: 1.0.0
License: MIT
"""

from torch import nn


class ModuleProxyWrapper(nn.Module):
    """
    A wrapper for `DistributedDataParallel` (DDP) modules that forwards requests for missing attributes
    and method calls to the underlying module wrapped by DDP (the twice-wrapped module).

    This class ensures that attributes and methods of the original module are accessible even when the
    module is wrapped in DDP. It also provides compatibility for `state_dict` and `load_state_dict` methods.

    Args:
        module (nn.Module): The module to wrap. Must be an instance of `DistributedDataParallel` or similar.

    Attributes:
        module (nn.Module): The wrapped module.
    """

    def __init__(self, module: nn.Module):
        """
        Initializes the `ModuleProxyWrapper` with the given module.

        Args:
            module (nn.Module): The module to wrap. Must have an attribute `module` representing the wrapped module.

        Raises:
            AssertionError: If the provided module does not have an attribute `module`.
        """
        super().__init__()
        assert hasattr(
            module, "module"
        ), "ModuleProxyWrapper expects input to wrap another module"
        self.module = module

    def __getattr__(self, name):
        """
        Forwards requests for missing attributes to the wrapped module or the twice-wrapped module.

        Args:
            name (str): The name of the attribute to retrieve.

        Returns:
            Any: The value of the requested attribute.

        Raises:
            AttributeError: If the attribute does not exist in the wrapped or twice-wrapped module.
        """
        try:
            # Defer to nn.Module's logic
            return super().__getattr__(name)
        except AttributeError:
            try:
                # Forward to the once-wrapped module
                return getattr(self.module, name)
            except AttributeError:
                # Forward to the twice-wrapped module
                return getattr(self.module.module, name)

    def state_dict(self, *args, **kwargs):
        """
        Forwards the `state_dict` call to the twice-wrapped module.

        Args:
            *args: Positional arguments for the `state_dict` method.
            **kwargs: Keyword arguments for the `state_dict` method.

        Returns:
            dict: The state dictionary of the twice-wrapped module.
        """
        if isinstance(self.module, nn.parallel.DistributedDataParallel):
            return self.module.module.state_dict(*args, **kwargs)
        else:
            return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        """
        Forwards the `load_state_dict` call to the twice-wrapped module.

        Args:
            *args: Positional arguments for the `load_state_dict` method.
            **kwargs: Keyword arguments for the `load_state_dict` method.

        Returns:
            Any: The result of the `load_state_dict` method call on the twice-wrapped module.
        """
        if isinstance(self.module, nn.parallel.DistributedDataParallel):
            return self.module.module.load_state_dict(*args, **kwargs)
        else:
            return self.module.load_state_dict(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """
        Forwards the `forward` call to the wrapped module.

        Args:
            *args: Positional arguments for the `forward` method.
            **kwargs: Keyword arguments for the `forward` method.

        Returns:
            Any: The result of the `forward` method call on the wrapped module.
        """
        return self.module(*args, **kwargs)
