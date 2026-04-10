# -*- coding: utf-8 -*-
from omegaconf import OmegaConf, DictConfig
import torch
import torch.nn as nn
from typing import Optional
import torchvision.transforms.functional as F

import src.libs.model.env.whisper_setup
from src.libs.model.env.whisper_setup import setup_whisper_flamingo_environment
from whisper.model import LayerNorm, Linear, MultiHeadAttention
import whisper
from whisper import log_mel_spectrogram
import whisper_flamingo_whisper as wfw
from src.libs.logger.log import getLogger


class CWhisperFlamingoBackbone(nn.Module):
    """Backbone wrapper for Whisper-Flamingo (audio-video encoder only)."""

    def __init__(self, config: dict | OmegaConf = None):
        super().__init__()
        self._logger = getLogger("CBackbone")
        self._logger.info("Setting up Whisper-Flamingo environment")
        setup_whisper_flamingo_environment()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(config, dict):
            config = OmegaConf.create(config)
        elif not isinstance(config, DictConfig):
            raise TypeError("Configuration must be a dict or OmegaConf DictConfig")

        self._config = config

        self._logger.info("Initializing Whisper-Flamingo backbone")
        self.output_dim = config.output_dim
        self.av_fusion = config.av_fusion
        self.av_fusion_method = config.get("av_fusion_method", "cross_attention")

        self.whisper_flamingo_model = self.load_whisper_flamingo_model()
        self.encoder = self.whisper_flamingo_model.encoder.to(self.device)
        # self.encoder.av_fusion = "lip-reader"  # ensure using lip-reader fusion
        self._delete_decoder()

        if self.av_fusion == "lip-reader" or self.av_fusion_method == "self_attention":
            n_state = config.n_state * 2
        else:
            n_state = config.n_state

        if self.av_fusion_method == "self_attention":
            self.self_attn = MultiHeadAttention(n_state, config.self_attention.n_head)
            self.self_attn_ln = LayerNorm(n_state)
        elif self.av_fusion_method == "cross_attention":
            self.cross_attn = MultiHeadAttention(n_state, config.cross_attention.n_head)
            self.cross_attn_ln = LayerNorm(n_state)
        elif self.av_fusion_method == "concat":
            pass

        self.mlp = nn.Sequential(
            Linear(n_state, config.cross_attention.n_mlp), nn.GELU(),
            Linear(config.cross_attention.n_mlp, n_state))
        self.mlp_ln = LayerNorm(n_state)
        self.proj_fusion = nn.Linear(n_state, self.output_dim).to(self.device)

        self.training = False
        if config.freeze:
            self._logger.info("Freezing Whisper-Flamingo parameters")
            for p in self.parameters():
                p.requires_grad = False
            self.encoder.eval()
        else:
            self.training = True

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
        log_mel_spectrogram = wfw.log_mel_spectrogram(audio_waveform).to(self.device, non_blocking=True)

        # [B, T, H, W, C] -> [B, C, T, H, W]
        video_frames = video_frames.float().permute((0, 4, 1, 2, 3)).contiguous()
        video_frames = video_frames.to(device=self.device, dtype=torch.float32, non_blocking=True)
        # Convert RGB to grayscale if needed
        if video_frames.size(1) == 3:
            video_frames = video_frames.mean(dim=1, keepdim=True)

        self._logger.dev(f"Forward Whisper-Flamingo: audio={log_mel_spectrogram.shape}, "
                         f"video={None if video_frames is None else video_frames.shape}")

        audio_embed, video_embed = self.encoder(x=log_mel_spectrogram, x_v=video_frames,
                                                training=self.training, padding_mask=None)
        if self.av_fusion_method == "cross_attention":
            # Cross-attention layers
            x = video_embed
            for layer_idx in range(self._config.cross_attention.n_layer):
                x = x + self.cross_attn(self.cross_attn_ln(x), audio_embed)[0]
                x = x + self.mlp(self.mlp_ln(x))
        else:
            x = audio_embed
            Ta = int(audio_embed.size(1))
            Tv = int(video_embed.size(1))
            if Ta == 2 * Tv:
                x = x.reshape(x.size(0), Tv, 2, x.size(2)).mean(dim=2)

            # Concatenate along features
            x = torch.cat((x, video_embed), dim=-1)
            if self.av_fusion_method == "self_attention":
                for layer_idx in range(self._config.self_attention.n_layer):
                    x = x + self.self_attn(self.self_attn_ln(x))[0]
                    x = x + self.mlp(self.mlp_ln(x))

        # Feature projection to final output dimension
        x = self.proj_fusion(x)  # [B, T, output_dim]
        return x

    def load_whisper_flamingo_model(self):
        self._logger.info(f"Loading Whisper Flamingo model type {self._config.model_type} on device {self._config.device}")
        whisper_model = wfw.load_model(self._config.model_type,
                                       device=self._config.device,
                                       download_root=self._config.download_root,
                                       video=True,
                                       video_model_path=self._config.video_model_path,
                                       av_hubert_path=self._config.av_hubert_path,
                                       av_hubert_encoder=self._config.use_av_hubert_encoder,
                                       av_fusion=self._config.av_fusion,
                                       add_adapter=True,
                                       add_gated_x_attn=1)

        self._logger.info(
            f"Whisper model loaded with {sum(p.numel() for p in whisper_model.parameters() if p.requires_grad)} trainable parameters")
        audio_model_path = self._config.get("audio_model_path", None)

        if audio_model_path is not None:
            self._logger.debug("Loading audio checkpoint")
            state_dict = torch.load(audio_model_path, map_location=self._config["device"])
            self._logger.debug(f"Models keys after loading {state_dict.keys()}")
            state_dict = state_dict['state_dict']
            state_dict_updated = {k[6:]: v for k, v in state_dict.items()}  # remove 'model.'
            try:  # newer models have learnable scaler init 1
                whisper_model.load_state_dict(state_dict_updated)
            except Exception as e:
                self._logger.error(str(e))
                self._logger.error("Loading weights with strict=False")
                whisper_model.load_state_dict(state_dict_updated, strict=False)

        if (torch.cuda.is_available()) and self._config.get("use_fp16", False):
            self._logger.info("Converting Whisper-Flamingo model to half precision")
            whisper_model.encoder.audio_model.half()
            whisper_model.encoder.video_projection_scalar.half()
            whisper_model.encoder.video_model.half()
            model_to_num_layers = {'small': 12, 'medium': 24, 'large-v2': 32}
            if self._config["av_fusion"] == 'separate':
                for i in range(model_to_num_layers[self._config["model_type"]]):
                    whisper_model.decoder.blocks[i].attn_gate.data = whisper_model.decoder.blocks[
                        i].attn_gate.half()
                    whisper_model.decoder.blocks[i].ff_gate.data = whisper_model.decoder.blocks[
                        i].ff_gate.half()
        return whisper_model

    def _delete_decoder(self):
        """Delete the decoder to save memory if not needed."""
        self._logger.info("Deleting Whisper-Flamingo decoder to save memory")
        # Remove the decoder to free VRAM
        if hasattr(self.whisper_flamingo_model, "decoder"):
            del self.whisper_flamingo_model.decoder
            torch.cuda.empty_cache()

        del self.whisper_flamingo_model
