from pathlib import Path
from typing import Dict, List, Optional

from src.libs.logger.log import getLogger


class CConfigMultimodalTurnTakingDataset:
    REQUIRED_KEYS = [
        "horizon",
        "sample_rate",
        "frame_hz",
        "mono",
        "num_pool",
        "use_cache",
        "multimodal",
        "use_face_encoder",
        "exclude_av_cache",
        "preload_av",
        "cache_dir"
    ]

    def __init__(self, cfg: Dict):
        self._logger = getLogger(self.__class__.__name__)
        self._validate(cfg)

        self.horizon: float = cfg["horizon"]
        self.sample_rate: int = cfg["sample_rate"]
        self.frame_hz: int = cfg["frame_hz"]
        self.mono: bool = cfg["mono"]
        self.num_pool: int = cfg["num_pool"]

        self.use_cache: bool = cfg["use_cache"]
        self.multimodal: bool = cfg["multimodal"]
        self.use_face_encoder: bool = cfg["use_face_encoder"]
        self.exclude_av_cache: bool = cfg["exclude_av_cache"]
        self.preload_av: bool = cfg["preload_av"]

        self.cache_dir: Path = Path(cfg["cache_dir"])

    def _validate(self, cfg: Dict):
        missing = [k for k in self.REQUIRED_KEYS if k not in cfg]
        if missing:
            raise ValueError(f"Missing required config keys: {missing}")

    def print_config(self):
        self._logger.info("=== Dataset Config ===")
        for k, v in self.__dict__.items():
            self._logger.info(f"{k}: {v}")
        self._logger.info("======================")

