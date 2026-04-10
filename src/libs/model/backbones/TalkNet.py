# -*- coding: utf-8 -*-
import sys
from pathlib import Path

base_path = Path(__file__).resolve().parent

from src.libs.utils import repo_root
talknet_repo_path = (Path(repo_root()) / 'external' / 'TalkNet-ASD').resolve()
model_path = talknet_repo_path / 'model'
sys.path.insert(0, str(talknet_repo_path))

import types
# Fake a 'model' package so 'model.talkNetModel' works
saved_model_pkg = sys.modules.get("model")
model_module = types.ModuleType('model')
model_module.__path__ = [str(model_path)]
sys.modules['model'] = model_module

from talkNet import talkNet

if saved_model_pkg is not None:
    sys.modules["model"] = saved_model_pkg
else:
    del sys.modules["model"]

from omegaconf import OmegaConf, DictConfig
import python_speech_features
import torch
from typing import Optional
import torchvision.transforms.functional as F

from src.libs.logger.log import getLogger


class CTalkNetBackbone(talkNet):
    """Wrapper for TalkNet-ASD backbone."""
    def __init__(self, config: dict | OmegaConf = None):
        super().__init__()
        self._logger = getLogger(self.__class__.__name__)

        if isinstance(config, dict):
            config = OmegaConf.create(config)
        elif not isinstance(config, DictConfig):
            raise TypeError("Configuration must be a dict or OmegaConf DictConfig")

        self.output_dim = config.output_dim
        self.use_video = config.use_video
        self._config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.av_fusion_method = config.get("av_fusion_method", None)

        if self._config.pretrained:
            self.loadParameters(self._config.model_path)

        if config.freeze:
            self._logger.info("Freezing TalkNet parameters")
            for p in self.parameters():
                p.requires_grad = False
            self.eval()

    def forward(
        self, audio_waveform: torch.Tensor, video_frames: Optional[torch.Tensor] = None) -> (
        torch.Tensor):
        """
        Args:
            audio_waveform: (B, T_audio) or (B, 1, T_audio)
            video_frames: (B, T_video, W, H) or (B, T_video, C, W, H)
                RGB input will be automatically converted to grayscale.
        Returns:
            fused_features: Tensor of shape (B*T, output_dim)
        """
        # Audio encoding
        self._logger.dev(f"Processing audio waveform with shape: {audio_waveform.shape}")
        mfcc_features = self._waveform_to_mfcc(audio_waveform)
        audio_embed = self.model.forward_audio_frontend(mfcc_features)

        if self.use_video and video_frames is not None:
            # Visual encoding
            if video_frames.ndim == 5:
                B, T, W, H, C = video_frames.shape
                video_frames = video_frames.permute(0, 1, 4, 3, 2).reshape(B * T, C, H, W)
                video_frames = F.rgb_to_grayscale(video_frames, num_output_channels=1)
                video_frames = video_frames.squeeze(1).reshape(B, T, H, W)
            elif video_frames.ndim == 4:
                # Already (B, T, W, H)
                pass

            self._logger.dev(f"Processing video frames with shape: {video_frames.shape}")
            visual_embed = self.model.forward_visual_frontend(video_frames)

            # Cross attention
            self._logger.dev(f"Shapes before cross attention: audio_embed: {audio_embed.shape}, "
                             f"visual_embed: {visual_embed.shape}")
            audio_embed, visual_embed = self.model.forward_cross_attention(
                audio_embed, visual_embed)
            self._logger.dev(f"Shapes after cross attention: audio_embed: {audio_embed.shape}, "
                             f"visual_embed: {visual_embed.shape}")

            # Fusion — keep(B, T, 256)
            fused_features = torch.cat((audio_embed, visual_embed), dim=2)  # (B, T, 256)
            fused_features = self.model.selfAV(src=fused_features, tar=fused_features)
        else:
            # Audio-only fusion
            fused_features = self.model.forward_audio_backend(audio_embed)  # (B*T, 128)

        self._logger.dev(f"Fused features shape: {fused_features.shape}")
        return fused_features

    def loadParameters(self, path):
        self._logger.info(f"Loading pretrained TalkNet model from {self._config.model_path}")
        selfState = self.state_dict()
        loadedState = torch.load(path, weights_only=False, map_location=self.device)
        for name, param in loadedState.items():
            origName = name;
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    self._logger.warning(f"{origName} is not in the model.")
                    continue
            if selfState[name].size() != loadedState[origName].size():
                error_param = "Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size())
                self._logger.error(error_param)
                sys.stderr.write(error_param)
                continue
            selfState[name].copy_(param)

    def _waveform_to_mfcc(
        self, audio_waveform: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        """
        Convert raw waveform batch into MFCC features.

        Args:
            audio_waveform: (B, T_audio) raw float32 waveform at sample_rate
            sample_rate: audio sampling rate (default 16 kHz)

        Returns:
            mfcc_padded: (B, T_mfcc, 13) tensor of MFCC features
        """
        mfcc_batch = []
        for wav in audio_waveform.detach().cpu().numpy():
            mfcc_feat = python_speech_features.mfcc(
                wav, samplerate=sample_rate, numcep=13, winlen=0.025, winstep=0.010)  # (T, 13)
            mfcc_batch.append(torch.from_numpy(mfcc_feat).float())

        # Pad sequences to the same length
        mfcc_padded = torch.nn.utils.rnn.pad_sequence(mfcc_batch, batch_first=True)
        return mfcc_padded.to(audio_waveform.device)
