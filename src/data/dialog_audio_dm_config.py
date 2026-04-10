# coding: UTF-8
from dataclasses import dataclass
from omegaconf import OmegaConf


@dataclass
class CDialogueAudioDMConfig:
    """
        Dialogue Audio data module configuration.

        Args:
            cfg_data: configuration file.
    """
    cfg_data: OmegaConf

    def __post_init__(self):
        self.datasets = self.cfg_data.data.datasets
        self.type = self.cfg_data.data.type
        self.sample_rate = self.cfg_data.data.sample_rate
        self.frame_hz = self.cfg_data.data.frame_hz
        self.audio_duration = self.cfg_data.data.audio_duration

        # Sliding Window Dataset
        self.audio_overlap = self.cfg_data.data.audio_overlap
        self.audio_normalize = self.cfg_data.data.audio_normalize

        # IPU Dataset
        self.audio_context_duration = self.cfg_data.events.metric.audio_context_duration
        self.ipu_min_time = self.cfg_data.events.metric.ipu_min_time
        self.ipu_pause_time = self.cfg_data.events.metric.ipu_pause_time

        # VAD
        self.vad = self.cfg_data.data.vad
        self.vad_hz = self.cfg_data.data.vad_hz
        self.vad_horizon = self.cfg_data.data.vad_horizon
        self.vad_history = self.cfg_data.data.vad_history
        self.vad_history_times = self.cfg_data.data.vad_history_times
        self.flip_channels = self.cfg_data.data.flip_channels

        # DataLoder
        self.batch_size = self.cfg_data.train.training_features.batch_size
        self.pin_memory = self.cfg_data.train.training_features.pin_memory
        self.num_workers = self.cfg_data.train.training_features.num_workers
        self.shuffle = self.cfg_data.train.training_features.shuffle

        # Label
        self.label_type = self.cfg_data.data.label_type
        self.bin_times = self.cfg_data.data.bin_times
        self.pre_frames = self.cfg_data.data.pre_frames
        self.threshold_ratio = self.cfg_data.data.threshold_ratio

        # Modalities
        self.use_face_only = self.cfg_data.data.use_face_only
        self.use_face_encoder = self.cfg_data.data.use_face_encoder

        self.keys = [
            "waveform",
            "waveform_user1",
            "waveform_user2",
            "vad",
            "vad_history",
            "gaze_user1",
            "au_user1",
            "pose_user1",
            "head_user1",
            "face_user1",
            "gaze_user2",
            "au_user2",
            "pose_user2",
            "head_user2",
            "face_user2",
            "label",
            ]
