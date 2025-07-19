"""
Registry Initialization Module

This module initializes the registry system for the Atria models package. It imports
and initializes the model registry from the `ModuleRegistry` class, making it
accessible as a module-level constant.

The registry system provides a centralized way to register and retrieve model
components throughout the application.

Constants:
    MODEL: Registry group for model components

Example:
    >>> from atria_models.registry import MODEL
    >>> # Register a new model
    >>> @MODEL.register()
    >>> class MyModel:
    ...     pass
    >>> # Get a registered model
    >>> model_cls = MODEL.get("my_model")

Dependencies:
    atria_registry.ModuleRegistry: Provides the main registry class
    atria_models.registry.module_registry: Provides registry initialization
    atria_models.registry.registry_groups: Provides ModelRegistryGroup class

Author: Atria Development Team
Date: 2025-07-10
Version: 1.2.0
License: MIT
"""

from atria_registry import ModuleRegistry

from atria_models.registry.module_registry import init_registry
from atria_models.registry.registry_groups import (
    ModelPipelineRegistryGroup,
    ModelRegistryGroup,
)

init_registry()

MODEL: ModelRegistryGroup = ModuleRegistry().MODEL
"""Registry group for models.

Used to register and manage model components, including custom models and
pre-trained models. This registry allows for easy retrieval and management of
model classes throughout the application.
"""

MODEL_PIPELINE: ModelPipelineRegistryGroup = ModuleRegistry().MODEL_PIPELINE
"""Registry group for model pipelines.
Used to register and manage model pipelines, which are sequences of models
that process data in stages. This registry allows for easy retrieval and management
of model pipeline classes throughout the application.
"""

__all__ = ["MODEL", "MODEL_PIPELINE"]
