"""Module Registry"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._registry_group import RegistryGroup


class ModuleRegistry:
    """
    A centralized registry for managing components in the Atria application.

    This singleton class organizes components into registry groups and provides
    methods for registering, retrieving, and storing configurations. It ensures
    that all components are properly registered and accessible throughout the
    application.

    Attributes:
        _registry_groups (dict): A dictionary mapping registry group names to their instances.
        _instance (ModuleRegistry): The singleton instance of the `ModuleRegistry` class.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        """
        Create or retrieve the singleton instance of the `ModuleRegistry` class.

        This method ensures that only one instance of the `ModuleRegistry` class
        exists throughout the application. It also initializes the configuration
        store with the `AtriaConfig` node.

        Args:
            *args: Positional arguments passed to the class constructor.
            **kwargs: Keyword arguments passed to the class constructor.

        Returns:
            ModuleRegistry: The singleton instance of the `ModuleRegistry` class.
        """
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """
        Initialize the registry groups and configuration store.

        This method sets up the registry groups and registers the `AtriaConfig`
        node in the Hydra configuration store.
        """

        self._registry_groups: dict[str, RegistryGroup] = {}

    def add_registry_group(self, name: str, registry_group: RegistryGroup):
        """
        Add a new registry group to the module registry.

        Args:
            name (str): The name of the registry group.
            registry_group (RegistryGroup): The registry group instance to add.

        Raises:
            ValueError: If a registry group with the same name already exists.
        """
        if name in self._registry_groups:
            return
        self._registry_groups[name] = registry_group

    def get_registry_group(self, name: str) -> RegistryGroup:
        """
        Retrieve a registry group by name.

        Args:
            name (str): The name of the registry group.

        Returns:
            RegistryGroup: The registry group corresponding to the given name.

        Raises:
            KeyError: If the registry group with the given name does not exist.
        """
        if name.upper() not in self._registry_groups:
            raise KeyError(
                f"Registry group '{name}' not found. Available groups: {list(self._registry_groups.keys())}"
            )
        return self._registry_groups[name.upper()]

    def dump_to_yaml(self) -> dict[str, list[str]]:
        for group_name, group in self._registry_groups.items():
            print(f"Registry Group: {group_name}")
            for module_name in group.dump():
                print(f"  Module: {module_name}")

    def __getattr__(self, name: str) -> RegistryGroup:
        """
        Retrieve a registry group as an attribute.

        Args:
            name (str): The name of the registry group.

        Returns:
            RegistryGroup: The registry group corresponding to the given name.

        Raises:
            AttributeError: If the registry group does not exist.
        """
        if name in self._registry_groups:
            return self._registry_groups[name]
        raise AttributeError(f"ModuleRegistry has no attribute '{name}'")

    def __getitem__(self, name: str) -> RegistryGroup:
        """
        Retrieve a registry group using dictionary-like access.

        Args:
            name (str): The name of the registry group.

        Returns:
            RegistryGroup: The registry group corresponding to the given name.

        Raises:
            KeyError: If the registry group with the given name does not exist.
        """
        return self.get_registry_group(name)
