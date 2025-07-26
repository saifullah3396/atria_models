from __future__ import annotations

from typing import TYPE_CHECKING

from atria_models.registry import MODEL

if TYPE_CHECKING:
    from torch import nn


@MODEL.register("layoutlmv3/token_classification")
def layoutlmv3_for_token_classification(num_labels: int, **kwargs) -> nn.Module:
    """
    Create a LayoutLMv3 model for token classification.

    Args:
        num_labels (int): The number of output labels for the model.
        **kwargs: Additional keyword arguments for the model.

    Returns:
        nn.Module: The LayoutLMv3 model configured for token classification.
    """
    from atria_models.models.layoutlmv3.modeling_layoutlmv3 import (
        LayoutLMv3ForTokenClassification,
    )

    return LayoutLMv3ForTokenClassification(num_labels=num_labels, **kwargs)


@MODEL.register("layoutlmv3/sequence_classification")
def layoutlmv3_for_sequence_classification(num_labels: int, **kwargs) -> nn.Module:
    """
    Create a LayoutLMv3 model for sequence classification.

    Args:
        num_labels (int): The number of output labels for the model.
        **kwargs: Additional keyword arguments for the model.

    Returns:
        nn.Module: The LayoutLMv3 model configured for sequence classification.
    """
    from atria_models.models.layoutlmv3.modeling_layoutlmv3 import (
        LayoutLMv3ForSequenceClassification,
    )

    return LayoutLMv3ForSequenceClassification(num_labels=num_labels, **kwargs)


@MODEL.register("layoutlmv3/question_answering")
def layoutlmv3_for_question_answering() -> nn.Module:
    """
    Create a LayoutLMv3 model for question answering.

    Args:
        num_labels (int): The number of output labels for the model.
        **kwargs: Additional keyword arguments for the model.

    Returns:
        nn.Module: The LayoutLMv3 model configured for question answering.
    """
    from atria_models.models.layoutlmv3.modeling_layoutlmv3 import (
        LayoutLMv3ForQuestionAnswering,
    )

    return LayoutLMv3ForQuestionAnswering()
