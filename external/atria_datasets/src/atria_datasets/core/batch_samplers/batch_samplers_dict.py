"""
Batch Samplers Dictionary Module

This module defines the `BatchSamplersDict` class, which provides a container
for managing batch samplers used during training and evaluation phases. It allows
users to store and access batch samplers for different stages of a machine learning
workflow.

Classes:
    - BatchSamplersDict: A container for managing training and evaluation batch samplers.

Dependencies:
    - functools.partial: For creating partially initialized batch samplers.
    - torch.utils.data.BatchSampler: Base class for batch samplers.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from typing import Any

from pydantic import BaseModel, field_validator


class BatchSamplersDict(BaseModel):
    """
    A container for managing batch samplers for training and evaluation.

    This class provides a simple interface for storing and accessing batch
    samplers used during the training and evaluation phases of a machine
    learning workflow.

    Attributes:
        train (Optional[partial[BatchSampler]]): The batch sampler used for training.
        evaluation (Optional[partial[BatchSampler]]): The batch sampler used for evaluation.
    """

    train: Any | None = None
    evaluation: Any | None = None

    @field_validator("train", "evaluation")
    @classmethod
    def validate_batch_sampler(cls, value):
        """
        Validates that the batch sampler is either None or a callable.

        Args:
            cls: The class being validated.
            value: The value to validate.

        Returns:
            The validated value.
        """

        if value is not None:
            from torch.utils.data import BatchSampler  # type: ignore[import-not-found]

            assert isinstance(value, BatchSampler), (
                f"Expected BatchSampler, got {type(value)}"
            )
        return value
