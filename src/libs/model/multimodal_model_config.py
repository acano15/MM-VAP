from omegaconf import OmegaConf

from src.libs.configuration.configuration_abstract import CBaseConfig
from src.libs.logger.log import getLogger


class CMultimodalVAPConfig(CBaseConfig):
    """Validates and wraps multimodal VAP configuration values."""

    REQUIRED_KEYS = [
        "use_deterministic", "events_loss_weight", "sample_rate", "frame_hz", "bin_times",
        "encoder_type", "pretrained_vap", "pretrained_cpc", "freeze_encoder", "load_pretrained",
        "only_feature_extraction", "dim", "channel_layers", "cross_layers", "num_heads", "dropout",
        "context_limit", "onishi", "dim_nonverbal", "dim_gaze", "dim_head", "dim_face", "dim_body",
        "face_only", "use_face_encoder", "pretrained_face_encoder", "dim_face_encoder", "mode",
        "multimodal", "context_limit_cpc_sec", "lid_classify", "lid_classify_num_class",
        "lid_classify_adversarial", "lang_cond", "pretrained_vap", "pretrained_cpc",
        "pretrained_face_encoder", "use_backbone", "backbone", "events_configuration",
    ]
    def __init__(self, conf: dict | OmegaConf):
        self._logger = getLogger(self.__class__.__name__)
        super().__init__(conf)

    def get_defaults(self) -> dict:
        pass
