from omegaconf import OmegaConf

from src.libs.configuration.configuration_abstract import CBaseConfig
from src.libs.logger.log import getLogger


class CEventConfig(CBaseConfig):
    """Event-specific configuration."""

    REQUIRED_KEYS = [
        "min_context_time",
        "metric_time",
        "metric_pad_time",
        "max_time",
        "frame_hz",
        "equal_hold_shift",
        "prediction_region_time",
        "sh_pre_cond_time",
        "sh_post_cond_time",
        "sh_prediction_region_on_active",
        "bc_pre_cond_time",
        "bc_post_cond_time",
        "bc_max_duration",
        "bc_negative_pad_left_time",
        "bc_negative_pad_right_time",
        "long_onset_region_time",
        "long_onset_condition_time",
    ]

    def __init__(self, conf: dict | OmegaConf):
        self._logger = getLogger(self.__class__.__name__)
        super().__init__(conf)

    def get_defaults(self) -> dict:
        return {
            "min_context_time": 3.0,
            "metric_time": 0.2,
            "metric_pad_time": 0.05,
            "max_time": 20,
            "frame_hz": 50,
            "equal_hold_shift": 1,
            "prediction_region_time": 0.5,
            "sh_pre_cond_time": 1.0,
            "sh_post_cond_time": 1.0,
            "sh_prediction_region_on_active": True,
            "bc_pre_cond_time": 1.0,
            "bc_post_cond_time": 1.0,
            "bc_max_duration": 1.0,
            "bc_negative_pad_left_time": 1.0,
            "bc_negative_pad_right_time": 2.0,
            "long_onset_region_time": 0.2,
            "long_onset_condition_time": 1.0,
        }
