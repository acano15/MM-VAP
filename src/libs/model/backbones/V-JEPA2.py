import os
import sys
from pathlib import Path
import types

base_path = Path(__file__).resolve().parent
sys.path.insert(0, str((base_path.parent.parent / 'logger').resolve()))
sys.path.insert(0, str((base_path.parent.parent / 'utils').resolve()))

from utils import repo_root
vjepa2_repo_path = (Path(repo_root()) / 'external' / 'VJEPA2' / 'src' / 'models').resolve()
sys.path.insert(0, str(vjepa2_repo_path))

from omegaconf import OmegaConf, DictConfig
import torch
from typing import Optional
import torchvision.transforms.functional as F
import torch.nn as nn

from models import AudioEncoder, VisionEncoder, FusionTransformer
from src.libs.logger.log import getLogger


class CVJepa2Backbone(nn.Module):
    """Wrapper for V-JEPA v2 multimodal backbone (audio + video)."""

    def __init__(self, config: dict | OmegaConf = None):
        super().__init__()
        self._logger = getLogger("CBackbone")
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        elif not isinstance(config, DictConfig):
            raise TypeError("Configuration must be a dict or OmegaConf DictConfig")

        self.output_dim = config.output_dim
        self.use_video = config.use_video
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # === Load pretrained encoders ===
        self.audio_encoder = AudioEncoder(pretrained=self._config.get("pretrained_audio", True)).to(self.device)
        self.visual_encoder = VisionEncoder(pretrained=self._config.get("pretrained_video", True)).to(self.device) \
                              if self.use_video else None
        self.fusion = FusionTransformer(dim=self.output_dim, num_heads=8, num_layers=6).to(self.device)

        # Projection layers (ensure consistent dim)
        self.audio_proj = nn.Linear(self.audio_encoder.output_dim, self.output_dim)
        if self.use_video:
            self.visual_proj = nn.Linear(self.visual_encoder.output_dim, self.output_dim)

        # Optional freeze
        if config.freeze:
            self._logger.info("Freezing V-JEPA2 backbone")
            for p in self.parameters():
                p.requires_grad = False
            self.eval()

    def forward(self, audio_waveform: torch.Tensor, video_frames: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            audio_waveform: (B, T_audio) or (B, 1, T_audio), 16 kHz waveform
            video_frames: (B, T_video, C, H, W) or (B, T_video, H, W, C), faces at 112x112
        Returns:
            fused_features: (B, T, output_dim)
        """

        # === Audio encoding ===
        if audio_waveform.ndim == 3:  # (B, 1, T) -> (B, T)
            audio_waveform = audio_waveform.squeeze(1)
        self._logger.debug(f"Audio waveform shape: {audio_waveform.shape}")

        audio_embed = self.audio_encoder(audio_waveform)         # (B, T_a, D_a)
        audio_embed = self.audio_proj(audio_embed)               # (B, T_a, output_dim)

        if self.use_video and video_frames is not None:
            # Reformat video to (B*T, C, H, W)
            if video_frames.ndim == 5 and video_frames.shape[-1] in [1, 3]:  # (B, T, H, W, C)
                video_frames = video_frames.permute(0, 1, 4, 2, 3)  # (B, T, C, H, W)
            B, T, C, H, W = video_frames.shape
            video_in = video_frames.reshape(B * T, C, H, W)

            # === Visual encoding ===
            visual_embed = self.visual_encoder(video_in)          # (B*T, D_v)
            visual_embed = visual_embed.view(B, T, -1)            # (B, T, D_v)
            visual_embed = self.visual_proj(visual_embed)         # (B, T, output_dim)

            # === Fusion ===
            fused_features = self.fusion(audio_embed, visual_embed)  # (B, T, output_dim)

        else:
            fused_features = audio_embed

        self._logger.debug(f"V-JEPA2 fused features: {fused_features.shape}")
        return fused_features
