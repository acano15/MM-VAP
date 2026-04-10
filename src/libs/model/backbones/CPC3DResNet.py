# -*- coding: utf-8 -*-
from omegaconf import OmegaConf, DictConfig
import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
from torchvision.models.video import r3d_18, R3D_18_Weights

from src.libs.logger.log import getLogger
from src.libs.model.encoder import EncoderCPC, EncoderFormerDFER


class CCPC3DResNetBackbone(nn.Module):
    """Backbone wrapper for Whisper-Flamingo (audio-video encoder only)."""

    def __init__(self, config: dict | OmegaConf = None):
        super().__init__()
        self._logger = getLogger("CBackbone")
        self._logger.info("Setting up Whisper-Flamingo environment")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(config, dict):
            config = OmegaConf.create(config)
        elif not isinstance(config, DictConfig):
            raise TypeError("Configuration must be a dict or OmegaConf DictConfig")

        self.output_dim = config.output_dim
        self._config = config
        self._logger.info("Initializing CCPC3DResNet backbone")

        if not self._config.pretrained:
            self._config.audio_model_path = None

        self.audio_encoder = EncoderCPC(
            cpc_model_pt=self._config.audio_model_path,
            load_pretrained=self._config.pretrained,
            freeze=self._config.freeze,
            lim_context_sec=self._config.context_limit_cpc_sec,
            frame_hz=self._config.frame_hz
            ).to(self.device)

        weights = R3D_18_Weights.DEFAULT if self._config.pretrained else None
        self.resnet3d = r3d_18(weights=weights)
        self.video_proj = nn.Linear(512, 256)

        self.av_embed_dim = 256
        self.concat_embed_dim = self.av_embed_dim * 2
        self.dropout = config.dropout
        self.av_fusion_method = config.get("av_fusion_method", "cross_attention")
        if self.av_fusion_method == "self_attention":
            n_state = self.concat_embed_dim
            self.self_attn = nn.MultiheadAttention(
                embed_dim=n_state,
                num_heads=config.self_attention.n_head,
                dropout=self.dropout,
                batch_first=True,
                )
            self.self_attn_ln = nn.LayerNorm(n_state)
            self.mlp = nn.Sequential(
                nn.Linear(n_state, config.self_attention.n_mlp),
                nn.GELU(),
                nn.Linear(config.self_attention.n_mlp, n_state),
                )
            self.mlp_ln = nn.LayerNorm(n_state)
        elif self.av_fusion_method == "cross_attention":
            n_state = self.av_embed_dim
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=n_state,
                num_heads=config.cross_attention.n_head,
                dropout=self.dropout,
                batch_first=True,
                )
            self.cross_attn_ln = nn.LayerNorm(n_state)

            self.mlp = nn.Sequential(
                nn.Linear(n_state, config.cross_attention.n_mlp),
                nn.GELU(),
                nn.Linear(config.cross_attention.n_mlp, n_state),
                )
            self.mlp_ln = nn.LayerNorm(n_state)
        elif self.av_fusion_method == "concat":
            n_state = self.concat_embed_dim
        else:
            raise ValueError(f"Unsupported av_fusion_method: {self.av_fusion_method}")

        self.proj_fusion = nn.Linear(n_state, self.output_dim).to(self.device)

        if config.freeze:
            self._logger.info("Freezing pretrained encoders")
            for p in self.audio_encoder.parameters():
                p.requires_grad = False
            for p in self.resnet3d.parameters():
                p.requires_grad = False

            self.audio_encoder.freeze()
            self.audio_encoder.eval()
            self.resnet3d.eval()

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
        if audio_waveform.dim() == 3 and audio_waveform.size(1) == 1:
            audio_waveform = audio_waveform.squeeze(1)  # (B, T_audio)

        audio_embed = self.audio_encoder(
            audio_waveform, only_feature_extractor=self._config.only_feature_extractor)
        # [B, T, H, W, C] -> [B, C, T, H, W]
        video_frames = video_frames.float().permute((0, 4, 1, 2, 3)).contiguous()
        video_frames = video_frames.to(device=self.device, dtype=torch.float32, non_blocking=True)
        self._logger.dev(f"Forward Whisper-Flamingo: audio={audio_waveform.shape}, "
                         f"video={None if video_frames is None else video_frames.shape}")

        video_embed = self._forward_video(video_frames=video_frames,
                                          target_num_steps=audio_embed.size(1))
        if self.av_fusion_method == "cross_attention":
            x = audio_embed
            for _ in range(self._config.cross_attention.n_layer):
                x_ln = self.cross_attn_ln(x)
                x = x + self.cross_attn(
                    x_ln,
                    video_embed,
                    video_embed,
                    need_weights=False,
                    )[0]
                x = x + self.mlp(self.mlp_ln(x))
        else:
            x = torch.cat((audio_embed, video_embed), dim=-1)
            if self.av_fusion_method == "self_attention":
                for layer_idx in range(self._config.self_attention.n_layer):
                    x_ln = self.self_attn_ln(x)
                    x = x + self.self_attn(x_ln, x_ln, x_ln, need_weights=False)[0]
                    x = x + self.mlp(self.mlp_ln(x))

        # Feature projection to final output dimension
        x = self.proj_fusion(x)  # [B, T, output_dim]
        return x

    def _forward_video(self, video_frames: torch.Tensor, target_num_steps: int) -> torch.Tensor:
        """Extract temporal video embeddings aligned to the audio sequence.

            Args:
                video_frames: Video tensor with shape [B, C, T, H, W].
                target_num_steps: Target temporal length for fusion.

            Returns:
                Video embeddings with shape [B, target_num_steps, 256].
        """
        # Backbone without avgpool/fc
        x = self.resnet3d.stem(video_frames)
        x = self.resnet3d.layer1(x)
        x = self.resnet3d.layer2(x)
        x = self.resnet3d.layer3(x)
        x = self.resnet3d.layer4(x)  # [B, 512, Tv, H', W']

        # Pool only spatial dimensions, keep time
        x = x.mean(dim=(-1, -2))  # [B, 512, Tv]

        # Align temporal length to the audio/CPC sequence
        if x.size(-1) != target_num_steps:
            x = F.adaptive_avg_pool1d(x, target_num_steps)  # [B, 512, target_num_steps]

        x = x.transpose(1, 2).contiguous()  # [B, T_audio, 512]
        x = self.video_proj(x)  # [B, T_audio, 256]
        return x
