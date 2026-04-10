from abc import ABC, abstractmethod
from omegaconf import OmegaConf, DictConfig
from typing import Any

from src.libs.logger.log import getLogger


class CBaseConfig(ABC):
    """Abstract base class for all configuration wrappers.

    Provides:
        - OmegaConf conversion from dict
        - Validation for required keys
        - Attribute and dict-style access

    All subclasses must define:
        - REQUIRED_KEYS (list[str])
        - get_defaults() method
    """

    REQUIRED_KEYS: list[str] = None  # must be overridden

    def __init_subclass__(cls, **kwargs):
        """Ensure subclasses define REQUIRED_KEYS and get_defaults()."""
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "REQUIRED_KEYS") or cls.REQUIRED_KEYS is None:
            raise TypeError(f"Class '{cls.__name__}' must define 'REQUIRED_KEYS' as a list of strings.")
        if not isinstance(cls.REQUIRED_KEYS, list):
            raise TypeError(f"Class '{cls.__name__}.REQUIRED_KEYS' must be of type list[str].")

    def __init__(self, conf: dict | OmegaConf):
        self._logger = getLogger(self.__class__.__name__)
        self._logger.debug(f"Config input: {conf}")

        # Ensure configuration is OmegaConf DictConfig
        if isinstance(conf, dict):
            conf = OmegaConf.create(conf)
        elif not isinstance(conf, DictConfig):
            raise TypeError("Configuration must be a dict or OmegaConf DictConfig")

        # Validate required keys
        self._validate(conf)

        # Store resolved configuration
        self._conf = conf

    def _validate(self, conf: DictConfig):
        """Validate that all REQUIRED_KEYS exist in configuration."""
        for key in self.REQUIRED_KEYS:
            if key not in conf:
                self._logger.error(f"Missing required config key: '{key}'")
                raise ValueError(f"Missing required config key: '{key}'")

    def __getattr__(self, name: str) -> Any:
        """Allow attribute-style access (cfg.param)."""
        if name == "_conf":
            return super().__getattribute__(name)
        return getattr(self._conf, name)

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access (cfg['param'])."""
        return self._conf[key]

    def __repr__(self):
        return f"{self.__class__.__name__}({OmegaConf.to_yaml(self._conf, resolve=True)})"

    @abstractmethod
    def get_defaults(self) -> dict:
        """Return default configuration values."""
        pass
