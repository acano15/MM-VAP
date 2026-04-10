import os
import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Tuple, List, Optional, Union
from rich.console import Console
from rich.table import Table
from omegaconf import OmegaConf, DictConfig
from tabulate import tabulate
import io
import contextlib

from .encoder import EncoderCPC, EncoderFormerDFER
from src.libs.events.objective import ObjectiveVAP
from .modules import GPT, GPTStereo, Linear, LinearStereo, TransformerLayer, TransformerConcatLayer
from .multimodal_model_config import CMultimodalVAPConfig
from src.libs.utils import (repo_root, everything_deterministic, vad_fill_silences, vad_omit_spikes,
                   torch_set_attr)
from src.libs.logger.log import getLogger


# Definitions
TRAIN = "train"
VAL = "val"
TEST = "test"

BIN_TIMES: list = [0.2, 0.4, 0.6, 0.8]


class CMultimodalVAP(nn.Module):
    """ Multimodal VAP model class """
    def __init__(self, conf: dict | OmegaConf = None):
        self._logger = getLogger(self.__class__.__name__)

        super().__init__()

        self._conf = CMultimodalVAPConfig(conf)
        if self._conf.use_deterministic:
            everything_deterministic()

        self.sample_rate = self._conf.sample_rate
        self.frame_hz = self._conf.frame_hz
        self.temp_elapse_time = []

        self.encoder = None
        self.face_encoder = None

        self.num_nonverbal = None
        self.dim_intermodal = None
        self.dim_interperson = None
        self.dim_face_encoder = None

        if self._conf.use_backbone:
            self.backbone = self._init_backbone()
        else:
            self.encoder = self._init_audio_models()

        if self._conf.use_face_encoder:
            self.decrease_dimension = self._init_decrease_dim()
            self._init_face_encoder()

        if self._conf.multimodal:
            self._init_multimodal_layers()

        if self._conf.onishi:
            self._init_onishi_models()

        if self._conf.onishi or self._conf.use_face_encoder or self._conf.use_backbone:
            self._init_intermodal_transformers()
            self._init_interperson_transformers()

        self._freeze_encoders()
        self._init_output_heads()
        self._init_messages()

    def _init_backbone(self):
        """Initializes and loads a pretrained multimodal backbone like TalkNet."""
        self._logger.debug("Initializing multimodal backbone")
        if self._conf.backbone.type.lower() == "talknet":
            from .backbones.TalkNet import CTalkNetBackbone
            self._logger.debug("Initializing TalkNet backbone")
            config_talknet = self._conf.backbone.TalkNet
            config_talknet.model_path = os.path.abspath(
                os.path.join(repo_root(), config_talknet.model_path))
            result = CTalkNetBackbone(config=config_talknet)
        elif self._conf.backbone.type.lower() == "whisperflamingo":
            from .backbones.WhisperFlamingo import CWhisperFlamingoBackbone
            result = CWhisperFlamingoBackbone(config=self._conf.backbone.WhisperFlamingo)
        elif self._conf.backbone.type.lower() == "cpc3dresnet":
            from .backbones.CPC3DResNet import CCPC3DResNetBackbone
            result = CCPC3DResNetBackbone(config=self._conf.backbone.CPC3DResNet)
            self._conf.backbone.freeze = self._conf.backbone.CPC3DResNet.freeze
        else:
            raise ValueError(f"Unknown backbone type: {self._conf.backbone.type}")

        if self._conf.backbone.freeze and self._conf.backbone.use_lora:
            from peft import LoraConfig, get_peft_model

            if self._conf.backbone.type.lower() == "talknet":
                lora_config = self._conf.backbone.TalkNet.LoRA
            elif self._conf.backbone.type.lower() == "whisperflamingo":
                lora_config = self._conf.backbone.WhisperFlamingo.LoRA
            elif self._conf.backbone.type.lower() == "cpc3dresnet":
                lora_config = self._conf.backbone.CPC3DResNet.LoRA
            else:
                raise ValueError(f"Unknown backbone type for LoRA config: {self._conf.backbone.type}")

            lora_cfg = LoraConfig(
                r=lora_config.rank,
                lora_alpha=lora_config.alpha,
                lora_dropout=lora_config.dropout,
                target_modules=lora_config.target_modules, )

            result = get_peft_model(result, lora_cfg)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                result.print_trainable_parameters()
            output_str = buf.getvalue().strip()
            self._logger.info(
                f"LoRA applied successfully: {output_str}")

        self._logger.debug("Multimodal backbone initialized")
        return result

    def _init_decrease_dim(self):
        enc = self._conf.encoder_type
        feat = self._conf.only_feature_extraction
        if enc == "cpc":
            pass
        elif enc == "wav2vec2":
            raise NotImplementedError("wav2vec2 encoder is not implemented yet")
        elif enc == "hubert":
            raise NotImplementedError("hubert encoder is not implemented yet")

        dim_map = {("wav2vec2", 1): 512, ("wav2vec2", 0): 1024, ("hubert_en_large", 1): 512,
                   ("hubert_en_large", 0): 1024}
        in_dim = dim_map.get(enc)
        return nn.Linear(in_dim, 256) if in_dim else None

    def _init_audio_models(self):
        if self._conf.encoder_type == "cpc":
            result = EncoderCPC(
                cpc_model_pt=self._conf.pretrained_cpc,
                load_pretrained=self._conf.load_pretrained == 1,
                freeze=self._conf.freeze_encoder,
                lim_context_sec=self._conf.context_limit_cpc_sec,
                frame_hz=self._conf.frame_hz
            )
        else:
            raise ValueError(f"Unknown encoder type: {self._conf.encoder_type}")

        self.ar_channel = GPT(
            dim=self._conf.dim,
            dff_k=3,
            num_layers=self._conf.channel_layers,
            num_heads=self._conf.num_heads,
            dropout=self._conf.dropout,
            context_limit=self._conf.context_limit,
        )
        self.ar = GPTStereo(
            dim=self._conf.dim,
            dff_k=3,
            num_layers=self._conf.cross_layers,
            num_heads=self._conf.num_heads,
            dropout=self._conf.dropout,
            context_limit=self._conf.context_limit,
        )
        return result

    def _init_onishi_models(self):
        features = {"face": self._conf.dim_face} if self._conf.face_only else {
            "gaze": self._conf.dim_gaze,
            "head": self._conf.dim_head,
            "face": self._conf.dim_face,
            "body": self._conf.dim_body
        }
        self._init_linear_channels(features)
        self._init_linear_stereo(features)
        self.num_nonverbal = len(features)
        self._init_onishi_transformers()

    def _init_linear_channels(self, features: dict):
        for name, dim_in in features.items():
            self._logger.debug(f"Initializing {name} channel linear")
            setattr(self, f"ar_channel_{name}", Linear(
                dim_in=dim_in,
                dim_out=self._conf.dim_nonverbal,
                dropout=self._conf.dropout
            ))

    def _init_linear_stereo(self, features: dict):
        for name in features:
            self._logger.debug(f"Initializing stereo linear for {name} channel")
            setattr(self, f"ar_{name}", LinearStereo(
                dim_in=self._conf.dim_nonverbal * 2,
                dim_out=self._conf.dim_nonverbal,
                dropout=self._conf.dropout
            ))

    def _init_onishi_transformers(self):
        self._logger.debug("Initializing Onishi-style transformers")
        dim = self._conf.dim_nonverbal * self.num_nonverbal
        self.nonverbal_transformer = TransformerConcatLayer(
            dim=dim,
            ffn_dim=dim * 3,
            num_heads=self._conf.num_heads,
            dropout=self._conf.dropout,
            context_limit=self._conf.context_limit
        )
        self.audio_nonverbal_transformer = TransformerConcatLayer(
            dim=dim + self._conf.dim,
            ffn_dim=dim * 3,
            num_heads=self._conf.num_heads,
            dropout=self._conf.dropout,
            context_limit=self._conf.context_limit
        )
        self._logger.debug("Onishi-style transformers initialized")

    def _init_face_encoder(self):
        self._logger.debug("Initializing face encoder")
        self.face_encoder = EncoderFormerDFER()
        if self._conf.load_pretrained:
            self.face_encoder.load_state_dict(
                torch.load(self._conf.pretrained_face_encoder, weights_only=True),
                strict=False
            )
            if self._conf.freeze_encoder:
                self.face_encoder.eval()

        # AR channel for face encoder output
        if self._conf.mode == 0:
            face_encoder_dim = self._conf.dim_face_encoder
        else:
            face_encoder_dim = self._conf.dim

        self.ar_channel_face_encoder = GPT(
            dim=face_encoder_dim,
            dff_k=3,
            num_layers=self._conf.channel_layers,
            num_heads=self._conf.num_heads,
            dropout=self._conf.dropout,
            context_limit=self._conf.context_limit,
            )

        # Compression is used in modes 1, 2, 3
        use_compress = self._conf.mode in {1, 2, 3}
        self._compress_face = None
        if use_compress:
            self.compress_face = Linear(
                dim_in=self._conf.dim_face_encoder,
                dim_out=self._conf.dim,
                dropout=self._conf.dropout
                )
            self.dim_face_encoder = self._conf.dim
        else:
            self.dim_face_encoder = self._conf.dim_face_encoder
        self._logger.debug("Face encoder initialized")

    def _init_multimodal_layers(self):
        # Add nonverbal channels depending on mode
        feature_map = {
            "gaze": self._conf.dim_gaze,
            "head": self._conf.dim_head,
            "body": self._conf.dim_body,
            "face": self._conf.dim_face if self._conf.mode == 3 else None,
            }
        for name, dim in feature_map.items():
            if dim:
                self._logger.debug(f"Initializing AR channel for {name} with dim {dim}")
                setattr(
                    self,
                    f"ar_channel_{name}",
                    Linear(dim_in=dim, dim_out=self._conf.dim_nonverbal, dropout=self._conf.dropout)
                    )

        self.num_nonverbal = sum(
            hasattr(self, f"ar_channel_{name}") for name in feature_map
            )

    def _init_intermodal_transformers(self):
        self._logger.debug("Initializing intermodal transformers")

        self.dim_intermodal = 0
        if self._conf.multimodal:
            self.dim_intermodal = self._conf.dim_nonverbal * self.num_nonverbal

        if self._conf.mode in {0, 2, 3}:
            if not self._conf.use_backbone:
                self.dim_intermodal += self._conf.dim
            else:
                self.dim_intermodal += self._conf.backbone_output_dim

            if self._conf.use_face_encoder:
                self.dim_intermodal += self.dim_face_encoder

            if self._conf.mode == 0:
                self.dim_interperson = self.dim_intermodal * 2
            else:
                self.dim_interperson = self.dim_intermodal

            self.intermodal_transformer = TransformerConcatLayer(
                dim=self.dim_intermodal,
                ffn_dim=self._conf.dim_nonverbal * self.num_nonverbal * 3,
                num_heads=self._conf.num_heads,
                dropout=self._conf.dropout,
                context_limit=self._conf.context_limit
                )
        elif self._conf.mode == 1:
            self.dim_intermodal = self.dim_interperson = self._conf.dim
            self.intermodal_GPT = GPTStereo(
                dim=self._conf.dim,
                dff_k=3,
                num_layers=self._conf.cross_layers,
                num_heads=self._conf.num_heads,
                dropout=self._conf.dropout,
                context_limit=self._conf.context_limit
                )

        self._logger.debug("Intermodal transformers initialized")

    def _init_interperson_transformers(self):
        self._logger.debug("Initializing interperson transformers")
        if self._conf.mode == 0:
            self.interperson_transformer = TransformerConcatLayer(
                dim=self.dim_interperson,
                ffn_dim=self._conf.dim_nonverbal * self.num_nonverbal * 3,
                num_heads=self._conf.num_heads,
                dropout=self._conf.dropout,
                context_limit=self._conf.context_limit)
        else:
            self.interperson_GPTStereo = GPTStereo(
                dim=self.dim_intermodal,
                dff_k=3,
                num_layers=self._conf.cross_layers,
                num_heads=self._conf.num_heads,
                dropout=self._conf.dropout,
                context_limit=self._conf.context_limit)

        self._logger.debug("Interperson transformers initialized")

    def _freeze_encoders(self):
        if self._conf.freeze_encoder:
            self._logger.debug("Freeze encoders")
            if self.encoder is not None:
                self.encoder.freeze()
                self.encoder.eval()

            if self.face_encoder is not None:
                self.face_encoder.freeze()
                self.face_encoder.eval()

    def _init_output_heads(self):
        self.objective = ObjectiveVAP(bin_times=self._conf.bin_times, frame_hz=self._conf.frame_hz)

        out_linear = 2
        if self._conf.onishi:
            out_linear = 1
            dim = self._conf.dim_nonverbal * self.num_nonverbal + self._conf.dim
        elif self._conf.use_face_encoder or self._conf.use_backbone:
            dim = self.dim_interperson
        else:
            dim = self._conf.dim

        self.va_classifier = nn.Linear(dim, out_linear)
        self.vap_head = nn.Linear(dim, self.objective.n_classes)

        if self._conf.lid_classify == 1:
            self.lid_classifier = nn.Linear(self._conf.dim, self._conf.lid_classify_num_class)
        elif self._conf.lid_classify == 2:
            self.lid_classifier_middle = nn.Linear(self._conf.dim * 2, self._conf.lid_classify_num_class)

        if self._conf.lang_cond == 1:
            self.lang_condition = nn.Linear(self._conf.lid_classify_num_class, self._conf.dim)

    def _init_messages(self):
        """Logs configuration details inferred from the forward pipeline."""
        branch = []
        features = []

        if self._conf.onishi:
            branch.append("ONISHI (nonverbal fusion)")
            features.extend(["face"] if self._conf.face_only else ["gaze", "head", "face", "body"])
            features.append("audio")
        elif self._conf.use_backbone or self._conf.use_face_encoder:
            if self._conf.use_backbone:
                branch.append(f"BACKBONE [{self._conf.backbone.type}]")
                features.extend(["audio", "video (face_im)"])
            if self._conf.use_face_encoder:
                branch.append("FACE ENCODER")
                features.extend(["audio", "face encoder (image)"])

            if self._conf.multimodal:
                features.extend(["gaze", "head", "body"])
                if self._conf.mode == 3:
                    features.append("raw face")
        else:
            branch.append("PURE AUDIO")
            features.append("audio")

        if getattr(self._conf, "lang_cond", 0) == 1:
            features.append("language conditioning")

        self._logger.dev("=== Multimodal Pipeline Configuration ===")
        self._logger.dev(f" Branch: {' + '.join(branch)}")
        self._logger.dev(f" Mode: {self._conf.mode}")
        if features:
            self._logger.dev(f" Features: {', '.join(sorted(set(features)))}")
        self._logger.dev("========================================")

    def encode_face(self, src1: Tensor, chunk_size=16):
        pad_size = chunk_size * math.ceil(src1.size(1) / chunk_size) - src1.size(1)
        padding = torch.zeros(src1.size(0), pad_size, src1.size(2), src1.size(3), src1.size(4)).to(
            self.device)
        concat_src = torch.concat([src1, padding], dim=1)
        concat_src = concat_src.view(
            -1, chunk_size, src1.size(2), src1.size(3), src1.size(4)).contiguous()
        x1 = self.face_encoder(concat_src)
        x1 = x1.view(src1.size(0), -1, self._conf.dim_face_encoder)
        x1 = x1[:, :src1.size(1), :].contiguous()
        return x1

    def forward(self,
                src: Union[Tensor, Dict[str, Union[Tensor, None]], None] = None,
                attention: bool = False,
                lang_info: list = None) -> Dict[str, Tensor]:
        """
        Forward pass of the multimodal VAP model

        Args:
            src (Tensor or dict): Input features or a dictionary of features.
            attention (bool): Whether to return attention weights.
            lang_info (list): Language information for language conditioning.
        Returns:
            Dict[str, Tensor]: Dictionary containing the model outputs.
        """
        ret = {}
        features_spkr1 = []
        features_spkr2 = []

        waveform = src.get("waveform")

        if self._conf.multimodal:
            gaze1 = src.get("gaze1")
            gaze2 = src.get("gaze2")
            head1 = src.get("head1")
            head2 = src.get("head2")
            face1 = src.get("face1")
            face2 = src.get("face2")
            body1 = src.get("body1")
            body2 = src.get("body2")

        if self._conf.use_backbone or self._conf.use_face_encoder:
            face_im1 = src.get("face_im1")
            face_im2 = src.get("face_im2")
            if self._conf.use_backbone:
                spkr1_main = self.backbone(audio_waveform=waveform[:, :1], video_frames=face_im1)
                spkr2_main = self.backbone(audio_waveform=waveform[:, 1:], video_frames=face_im2)
            else:
                x1 = self.encoder(
                    waveform[:, :1], only_feature_extractor=self._conf.only_feature_extraction)
                x2 = self.encoder(
                    waveform[:, 1:], only_feature_extractor=self._conf.only_feature_extraction)
                if self.decrease_dimension is not None:
                    x1 = torch.relu(self.decrease_dimension(x1))
                    x2 = torch.relu(self.decrease_dimension(x2))
                if self._conf.lang_cond == 1:
                    lang_info_data = torch.zeros(
                        x1.size(0), x1.size(1), self._conf.lid_classify_num_class).to(x1.device)
                    for b, lang in enumerate(lang_info):
                        lang_info_data[b, :, lang] = 1
                    x1 += self.lang_condition(lang_info_data)
                    x2 += self.lang_condition(lang_info_data)
                o1_audio = self.ar_channel(x1, attention=attention)["x"]
                o2_audio = self.ar_channel(x2, attention=attention)["x"]
                features_spkr1.append(o1_audio)
                features_spkr2.append(o2_audio)

            if self._conf.use_backbone:
                features_spkr1.append(spkr1_main)
                features_spkr2.append(spkr2_main)

            if self._conf.multimodal and self._conf.mode != 1:
                features_spkr1.append(self.ar_channel_gaze(gaze1)["x"])
                features_spkr1.append(self.ar_channel_head(head1)["x"])
                features_spkr1.append(self.ar_channel_body(body1)["x"])
                features_spkr2.append(self.ar_channel_gaze(gaze2)["x"])
                features_spkr2.append(self.ar_channel_head(head2)["x"])
                features_spkr2.append(self.ar_channel_body(body2)["x"])

            if self._conf.use_face_encoder:
                x1_face = self.encode_face(face_im1)
                x2_face = self.encode_face(face_im2)
                if self._conf.mode == 0:
                    o1_face = self.ar_channel_face_encoder(x1_face, attention=attention)["x"]
                    o2_face = self.ar_channel_face_encoder(x2_face, attention=attention)["x"]
                else:
                    o1_face_cp = self.compress_face(x1_face)["x"]
                    o1_face = self.ar_channel_face_encoder(o1_face_cp, attention=attention)["x"]
                    o2_face_cp = self.compress_face(x2_face)["x"]
                    o2_face = self.ar_channel_face_encoder(o2_face_cp, attention=attention)["x"]

                features_spkr1.append(o1_face)
                features_spkr2.append(o2_face)

            if self._conf.multimodal and self._conf.mode == 3:
                features_spkr1.append(self.ar_channel_face(face1)["x"])
                features_spkr2.append(self.ar_channel_face(face2)["x"])

            if self._conf.mode == 0:
                out1, _, _ = self.intermodal_transformer(features_spkr1)
                out2, _, _ = self.intermodal_transformer(features_spkr2)
                out_interperson, _, _ = self.interperson_transformer([out1, out2])
            elif self._conf.mode == 1:
                if self._conf.use_backbone:
                    out1 = spkr1_main
                    out2 = spkr2_main
                else:
                    out1 = self.intermodal_GPT(o1_audio, o1_face, attention=attention)["x"]
                    out2 = self.intermodal_GPT(o2_audio, o2_face, attention=attention)["x"]

                out_interperson = self.interperson_GPTStereo(out1, out2, attention=attention)["x"]
            elif self._conf.mode == 2 or self._conf.mode == 3:
                out1, _, _ = self.intermodal_transformer(features_spkr1)
                out2, _, _ = self.intermodal_transformer(features_spkr2)
                out_interperson = self.interperson_GPTStereo(out1, out2, attention=attention)["x"]

            vad_logits = self.va_classifier(out_interperson)
            logits = self.vap_head(out_interperson)
            ret["logits"] = logits
            ret["vad_logits"] = vad_logits
        elif self._conf.onishi:
            if self._conf.face_only:
                o1_face = self.ar_channel_face(face1)
                o2_face = self.ar_channel_face(face2)
                out_face = self.ar_face(o1_face["x"], o2_face["x"])
                out_nonverbal, _, _ = self.nonverbal_transformer([out_face["x"]])
            else:
                o1_gaze = self.ar_channel_gaze(gaze1)
                o1_head = self.ar_channel_head(head1)
                o1_face = self.ar_channel_face(face1)
                o1_body = self.ar_channel_body(body1)
                o2_gaze = self.ar_channel_gaze(gaze2)
                o2_head = self.ar_channel_head(head2)
                o2_face = self.ar_channel_face(face2)
                o2_body = self.ar_channel_body(body2)
                out_gaze = self.ar_gaze(o1_gaze["x"], o2_gaze["x"])
                out_head = self.ar_head(o1_head["x"], o2_head["x"])
                out_face = self.ar_face(o1_face["x"], o2_face["x"])
                out_body = self.ar_body(o1_body["x"], o2_body["x"])
                out_nonverbal, _, _ = self.nonverbal_transformer(
                    [out_gaze["x"], out_head["x"], out_face["x"], out_body["x"]]
                    )

            x1 = self.encoder(
                waveform[:, :1], only_feature_extractor=self._conf.only_feature_extraction)
            x2 = self.encoder(
                waveform[:, 1:], only_feature_extractor=self._conf.only_feature_extraction)
            o1 = self.ar_channel(x1, attention=attention)
            o2 = self.ar_channel(x2, attention=attention)
            out = self.ar(o1["x"], o2["x"], attention=attention)

            out_audio_nonverbal, _, _ = self.audio_nonverbal_transformer([out["x"], out_nonverbal])
            ret["logits"] = self.vap_head(out_audio_nonverbal)
            ret["vad_logits"] = self.va_classifier(out_audio_nonverbal)
        else:
            # Pure audio branch
            x1 = self.encoder(
                waveform[:, :1], only_feature_extractor=self._conf.only_feature_extraction)
            x2 = self.encoder(
                waveform[:, 1:], only_feature_extractor=self._conf.only_feature_extraction)
            if self.decrease_dimension is not None:
                x1 = torch.relu(self.decrease_dimension(x1))
                x2 = torch.relu(self.decrease_dimension(x2))
            o1 = self.ar_channel(x1, attention=attention)
            o2 = self.ar_channel(x2, attention=attention)
            out = self.ar(o1["x"], o2["x"], attention=attention)

            v1 = self.va_classifier(out["x1"])
            v2 = self.va_classifier(out["x2"])
            ret["vad_logits"] = torch.cat((v1, v2), dim=-1)
            ret["logits"] = self.vap_head(out["x"])

            if self._conf.lid_classify == 2:
                ret["lid"] = self.lid_classifier_middle(torch.cat((o1["x"], o2["x"]), dim=-1))
            if self._conf.lid_classify == 1:
                ret["lid"] = self.lid_classifier(out["x"])
            if attention:
                ret["self_attn"] = torch.stack([o1["attn"], o2["attn"]], dim=1)
                ret["cross_attn"] = out["cross_attn"]
                ret["cross_self_attn"] = out["self_attn"]

        return ret

    def summarize(self, max_depth: int = 1) -> str:
        """
        Pretty table summary of the model. Called when
        Trainer(enable_model_summary=True).
        Shows param counts, differentiates trainable vs frozen.
        """
        rows = []

        def count_params(module: nn.Module):
            total, trainable = 0, 0
            for p in module.parameters(recurse=True):  # include children
                num = p.numel()
                total += num
                if p.requires_grad:
                    trainable += num
            return total, trainable

        def human_format(num: int) -> str:
            if num >= 1e6:
                return f"{num / 1e6:.1f} M"
            elif num >= 1e3:
                return f"{num / 1e3:.1f} K"
            return str(num)

        def describe(module: nn.Module, name: str, depth: int, parent: str = ""):
            total, trainable = count_params(module)
            nontrainable = total - trainable
            mode = "train" if module.training else "eval"

            rows.append(
                [
                    f"{parent}{name}",
                    module.__class__.__name__,
                    human_format(total),
                    human_format(trainable),
                    human_format(nontrainable),
                    mode,
                    ])

            if depth < max_depth:
                for child_name, child in module.named_children():
                    describe(child, child_name, depth + 1, parent=f"{parent}{name}.")

        # kick off
        describe(self, self.__class__.__name__, depth=0)

        table = tabulate(
            [[i] + row for i, row in enumerate(rows)],
            headers=["", "Name", "Type", "Total Params", "Trainable", "Frozen", "Mode"],
            tablefmt="fancy_grid"
            )

        self._logger.debug("\n" + table)
        console = Console()

        rtable = Table(title=f"Model Summary: {self.__class__.__name__}")
        for col in ["#", "Name", "Type", "Total Params", "Trainable", "Frozen", "Mode"]:
            rtable.add_column(col, justify="right" if col in {"Total Params", "Trainable",
                                                              "Frozen"} else "left")

        for i, row in enumerate(rows):
            rtable.add_row(str(i), *row)

        console.print(rtable)
        return table
