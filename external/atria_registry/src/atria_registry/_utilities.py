"""Utility functions for the Atria Registry.

This module provides utility functions for writing configuration nodes to YAML files,
writing the module registry to YAML, and instantiating objects from configurations.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Any, get_args, get_origin

from atria_logger import get_logger
from pydantic import BaseModel
from pydantic_core import PydanticUndefined

logger = get_logger(__name__)


def write_config_nodes(
    root_package_dir: Path, d: dict, types: list[str] | None = None
) -> None:
    """
    Recursively writes configuration nodes to YAML files.

    This function traverses a dictionary of configuration nodes and writes each
    node to a YAML file in the specified directory. It skips Hydra-specific keys
    and ensures that directories are created as needed.

    Args:
        root_package_dir (Path): The root directory where configuration files will be written.
        d (dict): A dictionary containing configuration nodes.

    Returns:
        None
    """
    from hydra.core.config_store import ConfigNode
    from omegaconf import OmegaConf

    for key, node in d.items():
        if key in ["hydra", "_dummy_empty_config_.yaml"]:
            continue
        if isinstance(node, ConfigNode):
            if (
                node.group is not None
                and types is not None
                and not node.group.startswith(tuple(types))
            ):
                continue
            logger.info(f"Dumping configuration: {node.group}/{node.name}")
            node_file_path = (
                root_package_dir / node.group / node.name
                if node.group
                else root_package_dir / node.name
            )
            if not node_file_path.parent.exists():
                node_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(node_file_path, "w") as f:
                if node.package is not None:
                    f.write(f"# @package {node.package}\n")
                f.write(OmegaConf.to_yaml(node.node, sort_keys=False))
        elif isinstance(node, dict):
            write_config_nodes(root_package_dir, node, types=types)


def write_registry_to_yaml(
    config_root_dir: str, types: list[str] | None = None, delete_existing: bool = False
) -> None:
    """
    Writes the entire module registry to YAML files.

    This function registers all modules in the `ModuleRegistry`, retrieves the
    configuration store, and writes the configurations to YAML files in the
    specified directory.

    Args:
        config_dir (str): The directory where configuration files will be written.

    Returns:
        None
    """

    import shutil

    from hydra.core.config_store import ConfigStore

    from atria_registry._module_registry import ModuleRegistry

    if delete_existing:
        delete_confirmation = input(
            f"Are you sure you want to delete existing configurations in {config_root_dir}? (y/n): "
        )
        if delete_confirmation.lower() != "y":
            logger.info("Skipping deletion of existing configurations.")
            return
        if Path(config_root_dir).exists():
            shutil.rmtree(config_root_dir)
        Path(config_root_dir).mkdir(parents=True, exist_ok=True)

    cs: ConfigStore = ModuleRegistry().store
    root_package_dir = Path(config_root_dir)
    root_package_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Building atria configurations in {root_package_dir}")
    init_file = root_package_dir / "__init__.py"
    if not init_file.exists():
        init_file.touch()

    write_config_nodes(root_package_dir, cs.repo, types=types)


def instantiate_object_from_config(config, override_config: dict | None = None):
    from hydra_zen import instantiate
    from hydra_zen.third_party.pydantic import pydantic_parser
    from omegaconf import DictConfig, OmegaConf
    from omegaconf.errors import MissingMandatoryValue
    from rich.pretty import pretty_repr

    try:
        if override_config is not None:
            config = OmegaConf.merge(
                OmegaConf.create(config), OmegaConf.create(override_config)
            )
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)

        return instantiate(config, _convert_="object", _target_wrapper_=pydantic_parser)
    except MissingMandatoryValue as e:
        raise RuntimeError(
            f"Missing mandatory values in config:\n{pretty_repr(config, expand_all=True)}.\n"
            f"You can pass them as overrides overrides={{'key': 'value'}}."
        ) from e


def _get_package_base_path(package: str) -> str | None:
    """
    Retrieves the base path of the specified package.

    Args:
        package (str): The name of the package.

    Returns:
        str | None: The base path of the specified package as a string, or None if the package is not found.
    """
    spec = importlib.util.find_spec(package)  # type: ignore
    return str(Path(spec.origin).parent) if spec else None


def _resolve_module_from_path(module_path: str) -> type[Any | Callable[..., Any]]:
    """
    Resolves a class or function from a module path string.

    Args:
        module_path (str): The module path in the format 'module_name.class_name'.

    Returns:
        object: The resolved class or function.

    Raises:
        ValueError: If the module path is invalid or cannot be resolved.
    """
    path = module_path.rsplit(".", 1)
    if len(path) == 1:
        raise ValueError(
            f"Invalid module path: {module_path}. It should be in the form 'module_name.class_name'."
        )
    module_name, class_name = path
    module = import_module(module_name)
    return getattr(module, class_name)


def _get_parent_module(module_name: str) -> str:
    """
    Retrieves the parent module name from a given module name.

    Args:
        module_name (str): The name of the module.

    Returns:
        str: The parent module name. If the module has no parent, returns the module name itself.
    """
    return module_name.rsplit(".", 1)[0] if "." in module_name else module_name


def _convert_to_snake_case(s: str) -> str:
    """
    Converts a camel case string to snake case.

    Args:
        s (str): The camel case string.

    Returns:
        str: The snake case string (underscored and lowercase).
    """
    return re.sub(r"([A-Z])", r"_\1", s).lower().lstrip("_")


def _annotation_contains_submodel(tp: Any) -> bool:
    """
    Returns True if the annotation tp eventually contains a BaseModel subclass.
    Handles unions, optionals, containers, annotated, etc.
    """
    if tp is None:
        return False

    # Direct model class
    if isinstance(tp, type) and issubclass(tp, BaseModel):
        return True

    origin = get_origin(tp)
    args = get_args(tp)

    # No origin: not a container, nothing else to check
    if origin is None:
        return False

    # Check all args of unions/containers/etc.
    return any(_annotation_contains_submodel(arg) for arg in args)


def _extract_nested_defaults(model: type[BaseModel]) -> dict[str, Any]:
    defaults: dict[str, Any] = {}

    for name, field in model.model_fields.items():
        # Determine usable default
        if field.default is not PydanticUndefined:
            value = field.default
        else:
            continue  # no default defined

        # -----------------------------
        # Case 1: the field default is a BaseModel instance
        # -----------------------------
        if isinstance(value, BaseModel):
            defaults[name] = _extract_nested_defaults(value.__class__)
            continue

        # -----------------------------
        # Case 2: containers containing BaseModels
        # -----------------------------
        if isinstance(value, (list, tuple)):
            processed = []
            for item in value:
                if isinstance(item, BaseModel):
                    processed.append(_extract_nested_defaults(item.__class__))
                else:
                    processed.append(item)
            defaults[name] = processed
            continue

        if isinstance(value, dict):
            processed = {}
            for k, v in value.items():
                if isinstance(v, BaseModel):
                    processed[k] = _extract_nested_defaults(v.__class__)
                else:
                    processed[k] = v
            defaults[name] = processed
            continue

        # -----------------------------
        # Case 3: annotation says it's a model,
        # but the default is not an instance.
        # (This means the default is ill-formed or missing.)
        # -----------------------------
        if _annotation_contains_submodel(field.annotation):
            # We cannot instantiate the model (by your rules), so skip.
            continue

        # -----------------------------
        # Case 4: primitive default
        # -----------------------------
        defaults[name] = value

    return defaults


def _get_config_hash(params: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
