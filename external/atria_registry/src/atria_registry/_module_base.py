"""Registration base classes for Atria modules."""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from atria_logger import get_logger
from atria_types._utilities._repr import RepresentationMixin
from pydantic import BaseModel, ConfigDict

from atria_registry._utilities import _get_config_hash, _resolve_module_from_path

T_ModuleConfig = TypeVar("T_ModuleConfig", bound="ModuleConfig")

logger = get_logger(__name__)


class ModuleConfig(RepresentationMixin, BaseModel):
    """
    Base class for Atria module registry configurations.
    All registry configurations must inherit from this class.
    """

    __version__ = "0.0.0"
    __builds_with_kwargs__ = False
    model_config = ConfigDict(extra="forbid", frozen=True)
    module_path: str | None = None

    @property
    def hash(self) -> str | None:
        return _get_config_hash(self.model_dump())

    @property
    def kwargs(self) -> dict[str, Any]:
        return self.model_dump(exclude={"module_path"})

    def to_yaml(self) -> str:
        """Serialize the ModuleConfig to a YAML string."""
        from omegaconf import OmegaConf

        config_omegaconf = OmegaConf.create(self.model_dump())
        return OmegaConf.to_yaml(config_omegaconf)

    def build(self, **kwargs) -> Any:
        assert self.module_path is not None, (
            "module_path must be set to build the module."
        )
        module = _resolve_module_from_path(self.module_path)
        if isinstance(module, type):
            if self.__builds_with_kwargs__:
                current_kwargs = self.kwargs
                current_kwargs.update(kwargs)
                return module(**current_kwargs)
            else:
                return module(config=self, **kwargs)
        else:
            raise TypeError(
                f"Module at path {self.module_path} is neither a class nor a callable."
            )


class ConfigurableModule(RepresentationMixin, Generic[T_ModuleConfig]):
    """
    Base class for Atria modules that can be registered in the Atria registry.
    All modules that are to be registered must inherit from this class.
    """

    __config__: type[T_ModuleConfig]
    __abstract__: bool = True

    def __init__(self, config: T_ModuleConfig) -> None:
        assert isinstance(config, self.__config__)
        self._config: T_ModuleConfig = config

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Validate presence of Config at class definition time
        if hasattr(cls, "__abstract__") and cls.__abstract__:
            return

        if not hasattr(cls, "__config__"):
            raise TypeError(
                f"{cls.__name__} must define a nested `__config__` class inherited from ModuleConfig."
            )

        if not issubclass(cls.__config__, ModuleConfig):
            raise TypeError(
                f"{cls.__name__}.__config__ must subclass ModuleConfig. Got {cls.__config__} instead."
            )

    @property
    def config(self) -> T_ModuleConfig:
        return self._config


class PydanticConfigurableModule(RepresentationMixin, BaseModel):
    __version__ = "0.0.0"
    model_config = ConfigDict(extra="forbid", frozen=True)

    @property
    def kwargs(self) -> dict[str, Any]:
        return self.model_dump(exclude={"module_path"})

    def to_yaml(self) -> str:
        """Serialize the ModuleConfig to a YAML string."""
        from omegaconf import OmegaConf

        config_omegaconf = OmegaConf.create(self.model_dump())
        return OmegaConf.to_yaml(config_omegaconf)
