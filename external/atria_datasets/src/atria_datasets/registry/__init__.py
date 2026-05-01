"""Datasets registry initialization module

The registry system provides a centralized way to register and retrieve components
such as datasets, models, transformations, and pipelines throughout the application.

Attributes:
    DATASET: Registry group for dataset components
    DATA_PIPELINE: Registry group for data pipeline components
    BATCH_SAMPLER: Registry group for batch sampler components

Example:
    >>> from atria_registry import DATA_TRANSFORM, MODEL
    >>> # Register a new data transform
    >>> @DATA_TRANSFORM.register()
    >>> class MyTransform:
    ...     pass
    >>> # Get a registered model
    >>> model_cls = MODEL.get("my_model")
"""

from atria_datasets.registry.registry_groups import (
    BATCH_SAMPLER,
    DATA_PIPELINE,
    DATASET,
)

__all__ = ["DATASET", "DATA_PIPELINE", "BATCH_SAMPLER"]
