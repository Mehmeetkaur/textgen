"""Module Builder."""

import inspect
from collections.abc import Callable
from functools import partial
from inspect import isfunction
from typing import Any

from pydantic import ConfigDict

from ._module_base import ConfigurableModule, ModuleConfig
from ._utilities import _resolve_module_from_path


class ModuleBuilderConfig(ModuleConfig):
    """
    Configuration class for ModuleBuilder.
    """

    model_config = ConfigDict(extra="allow", frozen=True)
    module: type[Any] | Callable[..., Any]


class ModuleBuilder(ConfigurableModule[ModuleBuilderConfig]):
    __config__ = ModuleBuilderConfig

    def __init__(self, module: str | Any, **kwargs):
        self.module = (
            _resolve_module_from_path(module) if isinstance(module, str) else module
        )

        # extract defaults from module signature
        defaults = self._get_default_args(self.module)

        # merged kwargs: defaults < overridden kwargs
        merged = {**defaults, **kwargs}

        super().__init__(ModuleBuilderConfig(module=self.module, **merged))

    def _get_default_args(self, module: Any) -> dict[str, Any]:
        """Extract keyword args with defaults from __init__ or function signature."""

        # resolve signature (class → __init__)
        if inspect.isclass(module):
            sig = inspect.signature(module.__init__)
        else:
            sig = inspect.signature(module)

        defaults = {}

        for param in sig.parameters.values():
            # ignore positional-only and varargs and self
            if param.name == "self":
                continue
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            if param.default is inspect._empty:
                continue

            defaults[param.name] = param.default

        return defaults

    def __call__(self) -> Callable[..., Any]:
        """Returns an instance of the module with the provided configuration.

        Returns:
            Callable[..., Any]: An instance of the module.
        """
        kwargs = self.config.model_dump()
        if isfunction(self.module):
            return partial(self.module, **kwargs)
        return self.module(**kwargs)
