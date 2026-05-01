"""Exceptions related to dataset configurations."""


class ConfigurationNotFoundError(ValueError):
    """
    Exception raised when a requested dataset configuration is not found.

    Attributes:
        config_name (str): The name of the missing configuration.
        available_configs (list[str]): A list of available configurations.
    """

    def __init__(self, config_name: str, available_configs: list[str]):
        """
        Initializes the ConfigurationNotFoundError.

        Args:
            config_name (str): The name of the missing configuration.
            available_configs (list[str]): A list of available configurations.
        """
        super().__init__(
            f"Configuration '{config_name}' not found in the dataset. "
            f"Available configurations: {', '.join(available_configs)}"
        )


class SplitNotFoundError(Exception):
    pass
