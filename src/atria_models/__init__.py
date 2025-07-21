"""
Models Module

This module provides model-related utilities and components for the Atria framework. It includes model definitions, training interfaces, and utilities for managing neural network modules.

Submodules:
    - core: Core model implementations and utilities.
    - pipelines: Pipelines for specific model tasks.
    - utilities: Helper functions and classes for neural network modules.

Author: Saifullah
Date: April 14, 2025
"""

# ruff: noqa

from typing import TYPE_CHECKING

import lazy_loader as lazy

if TYPE_CHECKING:
    from atria_models.registry import (
        MODEL,  # noqa: F401 # Import the registry to ensure it is initialized
    )
    from atria_models.core.atria_model import AtriaModel
    from atria_models.core.diffusers_model import DiffusersModel
    from atria_models.core.local_model import LocalModel
    from atria_models.core.mmdet_model import MMDetModel
    from atria_models.core.timm_model import TimmModel
    from atria_models.core.torchvision_model import TorchHubModel
    from atria_models.core.transformers_model import (
        SequenceClassificationModel,
        TokenClassificationModel,
        ImageClassificationModel,
        QuestionAnsweringModel,
    )
    from atria_models.pipelines.atria_model_pipeline import AtriaModelPipeline
    from atria_models.pipelines.classification.image import ImageClassificationPipeline
    from atria_models.pipelines.classification.sequence import (
        SequenceClassificationPipeline,
    )
    from atria_models.pipelines.classification.token import TokenClassificationPipeline
    from atria_models.pipelines.qa.qa import QuestionAnsweringPipeline
    from atria_models.pipelines.autoencoding.image import AutoEncodingPipeline
    from atria_models.pipelines.classification.layout_token import (
        LayoutTokenClassificationPipeline,
    )
    from atria_models.pipelines.mmdet.object_detection import ObjectDetectionPipeline

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "core.atria_model": ["AtriaModel"],
        "core.diffusers_model": ["DiffusersModel"],
        "core.local_model": ["LocalModel"],
        "core.mmdet_model": ["MMDetModel"],
        "core.timm_model": ["TimmModel"],
        "core.torchvision_model": ["TorchHubModel"],
        "core.transformers_model": [
            "SequenceClassificationModel",
            "TokenClassificationModel",
            "ImageClassificationModel",
            "QuestionAnsweringModel",
        ],
        "pipelines.atria_model_pipeline": ["AtriaModelPipeline"],
        "pipelines.classification.image": ["ImageClassificationPipeline"],
        "pipelines.classification.sequence": ["SequenceClassificationPipeline"],
        "pipelines.classification.token": ["TokenClassificationPipeline"],
        "pipelines.qa.qa": ["QuestionAnsweringPipeline"],
        "pipelines.autoencoding.image": ["AutoEncodingPipeline"],
        "pipelines.classification.layout_token": ["LayoutTokenClassificationPipeline"],
        "pipelines.mmdet.object_detection": ["ObjectDetectionPipeline"],
    },
)
