# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

base_path = Path(__file__).resolve().parent
sys.path.insert(0, str(base_path.parent))
sys.path.insert(0, str((base_path.parent.parent / 'logger').resolve()))
sys.path.insert(0, str((base_path.parent.parent / 'utils').resolve()))
sys.path = list(dict.fromkeys(sys.path))

import numpy as np
import torch
import yaml
import hydra
from omegaconf import DictConfig

from env.whisper_setup import setup_whisper_flamingo_environment
from whisper_flamingo_utils import load_video_feats
import whisper
import whisper_flamingo_whisper as wfw
from log import getLogger, load_logger_config
from resolvers import OmegaConf
from util import repo_root


def load_whisper_flamingo_model(config):
    global logger

    logger.info(f"Loading Whisper Flamingo model type {config['model_type']} on device {config['device']}")
    whisper_model: wfw.Whisper = wfw.load_model(config["model_type"],
                                        device=config["device"],
                                        download_root=config["download_root"],
                                        video=True,
                                        video_model_path=config["video_model_path"],
                                        av_hubert_path=config["av_hubert_path"],
                                        av_hubert_encoder=config["use_av_hubert_encoder"],
                                        av_fusion=config["av_fusion"],
                                        add_adapter=True,
                                        add_gated_x_attn=1
                                        )
    logger.info(f"Whisper model loaded with {sum(p.numel() for p in whisper_model.parameters() if p.requires_grad)} trainable parameters")
    audio_model_path = config.get("audio_model_path", None)

    if audio_model_path is not None:
        logger.debug("Loading audio checkpoint")
        state_dict = torch.load(audio_model_path, map_location=config["device"])
        logger.debug(f"Models keys after loading {state_dict.keys()}")
        state_dict = state_dict['state_dict']
        state_dict_updated = {k[6:]: v for k, v in state_dict.items()}  # remove 'model.'
        try:  # newer models have learnable scaler init 1
            whisper_model.load_state_dict(state_dict_updated)
        except Exception as e:
            logger.error(str(e))
            logger.error("Loading weights with strict=False")
            whisper_model.load_state_dict(state_dict_updated, strict=False)

    if (torch.cuda.is_available()) and config["use_av_hubert_encoder"] == 1:
        whisper_model.encoder.video_projection_scalar.half()
        whisper_model.encoder.video_model.half()
        model_to_num_layers = {'small': 12, 'medium': 24, 'large-v2': 32}
        if config["av_fusion"] == 'separate':
            for i in range(model_to_num_layers[config["model_type"]]):
                whisper_model.decoder.blocks[i].attn_gate.data = whisper_model.decoder.blocks[i].attn_gate.half()
                whisper_model.decoder.blocks[i].ff_gate.data = whisper_model.decoder.blocks[i].ff_gate.half()
    return whisper_model


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg_dict: DictConfig) -> None:
    global logger
    setup_whisper_flamingo_environment()

    OmegaConf.resolve(cfg_dict)

    load_logger_config(cfg_dict.logger)
    logger = getLogger(name="Main")
    logger.info("Starting WhisperFlamingo test")


    lang = 'es'
    file_path = Path(repo_root()) / 'data' / 'test' / 'hey_haru1.mkv'

    logger.info("Loading Whisper flamingo")
    logger.debug(f"Config: {cfg_dict.model.backbone.WhisperFlamingo}")
    whisper_model: wfw.Whisper = load_whisper_flamingo_model(cfg_dict.model.backbone.WhisperFlamingo)

    logger.info(f"Processing video file {str(file_path)}")
    audio = wfw.load_audio(str(file_path.with_suffix('.wav')))
    mel = wfw.log_mel_spectrogram(audio).to(whisper_model.device)
    mel = mel.unsqueeze(0)
    
    video = load_video_feats(str(file_path), train=False)
    video = torch.tensor(video.astype(np.float32))
    video = video.unsqueeze(0).permute((0, 4, 1, 2, 3)).contiguous()  # [B, T, H, W, C] -> [B, C, T, H, W]
    video = video.half().to(whisper_model.device) if torch.cuda.is_available() else video
    
    logger.dev(f'audio shape and type {audio.shape} {audio.dtype}')
    logger.dev(f'audio mel shape and type {mel.shape} {mel.dtype}')
    logger.dev(f'video shape and type {video.shape} {video.dtype}')
    
    whisper_model.eval()  # AV-HuBERT batch norm and dropout

    pred = whisper_model.encoder(x=mel, x_v=video)
    logger.debug(f"Encoder output shapes: {pred[0].shape}, {pred[1].shape}")

    options = whisper.DecodingOptions(fp16=True if torch.cuda.is_available() else False, language=lang, without_timestamps=True, beam_size=1)
    pred = whisper_model.decode(mel, options, video)[0].text
    logger.info(f"Predicted text: {pred}")


if __name__ == '__main__':
    main()
