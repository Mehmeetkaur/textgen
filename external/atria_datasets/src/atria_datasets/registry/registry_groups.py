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

from atria_registry import ModuleRegistry, RegistryGroup

from atria_datasets.core.dataset._common import DatasetConfig


class DatasetRegistryGroup(RegistryGroup[DatasetConfig]):
    def load_module_config(self, module_path: str, **kwargs) -> DatasetConfig:
        """Dynamically load all registered modules in the registry group."""
        config = super().load_module_config(module_path, **kwargs)
        assert isinstance(config, DatasetConfig)
        return config


ModuleRegistry().add_registry_group(
    name="DATASET",
    registry_group=DatasetRegistryGroup(name="dataset", package="atria_datasets"),
)
ModuleRegistry().add_registry_group(
    name="DATA_PIPELINE",
    registry_group=RegistryGroup(name="data_pipeline", package="atria_datasets"),
)
ModuleRegistry().add_registry_group(
    name="BATCH_SAMPLER",
    registry_group=RegistryGroup(name="batch_sampler", package="atria_datasets"),
)


DATASET: DatasetRegistryGroup = ModuleRegistry().DATASET
"""Registry group for datasets.

Used to register and manage dataset-related components throughout the application.
Provides methods to register new datasets and retrieve existing ones by name.
"""

DATA_PIPELINE = ModuleRegistry().DATA_PIPELINE
"""Registry group for data pipelines.

Used to register and manage data pipeline components that handle data processing
workflows and transformations.
"""

BATCH_SAMPLER = ModuleRegistry().BATCH_SAMPLER
"""Registry group for batch samplers.

Used to register and manage batch sampling strategies that determine how data
is grouped into batches during training and inference.
"""
