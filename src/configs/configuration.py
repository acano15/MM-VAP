import os
from types import SimpleNamespace
from typing import Dict, Any

from src.libs.utils import repo_root


class CBaseConfig:
    """Wrapper class that builds a SimpleNamespace from cfg_dict."""

    def __init__(self, a_cfg_dict):
        pretrained_vap_path = os.path.abspath(os.path.join(repo_root(), a_cfg_dict.model.vap.model_path))
        pretrained_cpc_path = os.path.abspath(os.path.join(
            repo_root(), a_cfg_dict.model.encoder.cpc.model_path))
        pretrained_face_encoder_path = os.path.abspath(
            os.path.join(repo_root(),a_cfg_dict.model.non_verbal_cond.face.model_path))

        self._ns = SimpleNamespace(
            use_deterministic=a_cfg_dict.train.training_features.use_deterministic,
            events_loss_weight=a_cfg_dict.train.training_features.events_loss_weight,
            sample_rate=a_cfg_dict.data.sample_rate,
            frame_hz=a_cfg_dict.model.encoder.frame_hz,
            mono=a_cfg_dict.data.audio_mono,
            encoder_type=a_cfg_dict.model.encoder.name,
            wav2vec_type=a_cfg_dict.model.encoder.wav2vec.type,
            hubert_model=a_cfg_dict.model.encoder.hubert.type,
            pretrained_vap=pretrained_vap_path,
            pretrained_cpc=pretrained_cpc_path,
            va_history=a_cfg_dict.data.va_history,
            va_history_bins=a_cfg_dict.data.va_history_bins,
            bin_times=a_cfg_dict.data.bin_times,
            freeze_encoder=a_cfg_dict.model.audio_cond.freeze_encoder,
            load_pretrained=a_cfg_dict.model.encoder.pretrained,
            only_feature_extraction=False,
            dim=a_cfg_dict.model.audio_module.input_size,
            channel_layers=a_cfg_dict.model.encoder.output_layer,
            cross_layers=a_cfg_dict.model.encoder.cross_layers,
            num_heads=a_cfg_dict.model.model_kwargs.Transformer.num_heads,
            dropout=a_cfg_dict.model.audio_module.dropout,
            context_limit=a_cfg_dict.events.metric.min_context,
            onishi=a_cfg_dict.train.training_features.onishi,
            dim_nonverbal=a_cfg_dict.model.non_verbal_module.input_size,
            dim_gaze=a_cfg_dict.model.non_verbal_cond.gaze.input_size,
            dim_head=a_cfg_dict.model.non_verbal_cond.head.input_size,
            dim_face=a_cfg_dict.model.non_verbal_cond.au.input_size,
            dim_body=a_cfg_dict.model.non_verbal_cond.pose.input_size,
            face_only=a_cfg_dict.train.training_features.use_face_only,
            use_face_encoder=a_cfg_dict.train.training_features.use_face_encoder,
            pretrained_face_encoder=pretrained_face_encoder_path,
            dim_face_encoder=a_cfg_dict.model.non_verbal_cond.face.dim_face_encoder,
            mode=a_cfg_dict.train.training_features.mode,
            multimodal=a_cfg_dict.data.multimodal,
            context_limit_cpc_sec=-1,
            lid_classify=0,
            lid_classify_num_class=3,
            lid_classify_adversarial=0,
            lang_cond=False,

            # Backbone config
            use_backbone=a_cfg_dict.train.training_features.use_backbone,
            use_lora=a_cfg_dict.model.backbone.use_lora,
            backbone_type=a_cfg_dict.model.backbone.type,
            backbone_freeze=a_cfg_dict.model.backbone.freeze,
            backbone_output_dim=a_cfg_dict.model.backbone.output_dim,
            use_backbone_video=a_cfg_dict.model.backbone.use_video,
            backbone_talknet=a_cfg_dict.model.backbone.TalkNet,
            backbone_whisper_flamingo=a_cfg_dict.model.backbone.WhisperFlamingo,

            min_context_time=a_cfg_dict.events.metric.min_context,
            metric_time=a_cfg_dict.events.metric.onset_dur,
            metric_pad_time=a_cfg_dict.events.metric.pad,
            max_time=a_cfg_dict.events.metric.max_time,
            equal_hold_shift=a_cfg_dict.events.metric.equal_hold_shift,
            prediction_region_time=a_cfg_dict.events.metric.prediction_region_time,

            # Shift/Hold,
            sh_pre_cond_time=a_cfg_dict.events.SH.pre_offset_shift,
            sh_post_cond_time=a_cfg_dict.events.SH.post_onset_hold,
            sh_prediction_region_on_active=a_cfg_dict.events.SH.prediction_region_on_active,

            # Backchannel,
            bc_pre_cond_time=a_cfg_dict.events.BC.pre_silence_frames,
            bc_post_cond_time=a_cfg_dict.events.BC.post_silence_frames,
            bc_max_duration=a_cfg_dict.events.BC.max_duration_frames,
            bc_negative_pad_left_time=a_cfg_dict.events.BC.negative_pad_left_time,
            bc_negative_pad_right_time=a_cfg_dict.events.BC.negative_pad_right_time,

            # Long/Short,
            long_onset_region_time=a_cfg_dict.events.LS.onset_region_time,
            long_onset_condition_time=a_cfg_dict.events.LS.onset_condition_time
            )

    def get_namespace(self) -> SimpleNamespace:
        return self._ns

    def get_dict(self) -> Dict[str, Any]:
        """Return a recursive dict representation compatible with OmegaConf."""

        return self._to_dict(self._ns)

    def __getattr__(self, name):
        if name == "_ns":
            return super().__getattribute__(name)
        return getattr(self._ns, name)

    def __repr__(self):
        return repr(self._ns)

    def _to_dict(self, obj):
        if isinstance(obj, SimpleNamespace):
            return {k: self._to_dict(v) for k, v in vars(obj).items()}
        elif isinstance(obj, dict):
            return {k: self._to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._to_dict(v) for v in obj]
        else:
            return obj