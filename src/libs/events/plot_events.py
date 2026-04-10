import matplotlib as mpl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only
#from  utils import (get_activity_history, vad_list_to_onehot, load_waveform, get_audio_info,
#                    time_to_frames)
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
import json

mpl.use("agg")

from src.libs.logger.log import getLogger



class CPlotsEvents(Callback):
    def __init__(
        self,
        audio_mix_path,
        audio_user1_path,
        audio_user2_path,
        vad_json_path,
        text_json_path=None,
        save_dir="viz_events",
        plot_every_n_epochs=1
    ):
        self.audio_mix_path = audio_mix_path
        self.audio_user1_path = audio_user1_path
        self.audio_user2_path = audio_user2_path
        self.vad_json_path = vad_json_path
        self.text_json_path = text_json_path
        self.save_dir = Path(save_dir)
        self.plot_every_n_epochs = plot_every_n_epochs
        self._logger = getLogger(self.__class__.__name__)

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.plot_every_n_epochs != 0:
            return

        cfg_dict = pl_module.cfg_dict
        model = pl_module.eval()
        model.to(cfg_dict["train"]["device"])

        vad_list = self._load_json(self.text_json_path) if self.text_json_path else self._load_json(self.vad_json_path)
        text_list = self._load_json(self.text_json_path) if self.text_json_path else None

        sample = self._prepare_sample(cfg_dict, vad_list)
        sample = self._pad_sample(cfg_dict, sample)
        sample = self._to_device(sample, cfg_dict["train"]["device"])

        probs = self._run_model(model, sample, cfg_dict)
        inp = self._reconstruct_waveforms(cfg_dict, text_list, vad_list)

        from turntaking.visualization import plot_origin
        fig, _ = plot_origin(
            probs["p"],
            probs["bc_prediction"],
            sample=inp,
            sample_rate=cfg_dict["data"]["sample_rate"]
        )
        filename = f"epoch_{epoch}_{datetime.datetime.now().strftime('%H_%M_%S')}.png"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.save_dir / filename, dpi=300)
        plt.close(fig)
        self._logger.info(f"Saved VAP event plot to {self.save_dir / filename}")

    def _load_json(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def _prepare_sample(self, cfg_dict, vad_list):
        sr = cfg_dict["data"]["sample_rate"]
        normalize = True
        mono = True
        vad_hz = cfg_dict["data"]["vad_hz"]
        vad_hop_time = 1.0 / vad_hz
        vad_history_frames = (torch.tensor([60, 30, 10, 5]) / vad_hop_time).long().tolist()

        ret = {
            "waveform": load_waveform(self.audio_mix_path, sr, normalize, mono)[0],
            "waveform_user1": load_waveform(self.audio_user1_path, sr, normalize, mono)[0],
            "waveform_user2": load_waveform(self.audio_user2_path, sr, normalize, mono)[0]
        }

        duration = get_audio_info(self.audio_mix_path)["duration"]
        end_frame = time_to_frames(duration, vad_hop_time)
        all_vad = vad_list_to_onehot(vad_list, hop_time=vad_hop_time, duration=duration, channel_last=True)
        ret["vad"] = all_vad[:end_frame].unsqueeze(0)

        vad_history, _ = get_activity_history(all_vad, bin_end_frames=vad_history_frames, channel_last=True)
        ret["vad_history"] = vad_history[:end_frame][..., 0].unsqueeze(0)
        return ret

    def _pad_sample(self, cfg_dict, sample):
        vad_frame_num = int(cfg_dict["data"]["vad_hz"] * cfg_dict["data"]["audio_duration"])
        audio_frame_num = int(cfg_dict["data"]["sample_rate"] * cfg_dict["data"]["audio_duration"])

        sample["vad"] = torch.cat([sample["vad"], torch.zeros(1, vad_frame_num, 2)], dim=1)
        sample["vad_history"] = torch.cat([sample["vad_history"], torch.zeros(1, vad_frame_num, 5)], dim=1)
        for key in ["waveform", "waveform_user1", "waveform_user2"]:
            sample[key] = torch.cat([sample[key], torch.zeros(1, audio_frame_num)], dim=1)
        return sample

    def _to_device(self, sample, device):
        return {k: v.to(device) for k, v in sample.items()}

    def _run_model(self, model, sample, cfg_dict):
        vad_frame_num = int(cfg_dict["data"]["vad_hz"] * cfg_dict["data"]["audio_duration"])
        audio_frame_num = int(cfg_dict["data"]["sample_rate"] * cfg_dict["data"]["audio_duration"])
        ratio = audio_frame_num / vad_frame_num

        probs = []
        for i in range(0, sample["vad"].shape[1] - vad_frame_num):
            ret = {
                "vad": sample["vad"][:, i:i + vad_frame_num, :],
                "vad_history": sample["vad_history"][:, i:i + vad_frame_num, :],
                "waveform": sample["waveform"][:, int(i * ratio):int((i + vad_frame_num) * ratio)],
                "waveform_user1": sample["waveform_user1"][:, int(i * ratio):int((i + vad_frame_num) * ratio)],
                "waveform_user2": sample["waveform_user2"][:, int(i * ratio):int((i + vad_frame_num) * ratio)],
            }
            out = model.output(ret)
            probs.append(out["logits_vp"])
        logits = torch.cat(probs, dim=0).permute(1, 0, 2)
        return model.VAP(logits=logits, va=sample["vad"][:, vad_frame_num:].cpu())

    def _reconstruct_waveforms(self, cfg_dict, text_list, vad_list):
        return {
            "waveform": load_waveform(self.audio_mix_path, sample_rate=cfg_dict["data"]["sample_rate"])[0],
            "waveform_trainer": load_waveform(self.audio_user1_path, sample_rate=cfg_dict["data"]["sample_rate"])[0],
            "waveform_trainee": load_waveform(self.audio_user2_path, sample_rate=cfg_dict["data"]["sample_rate"])[0],
            "vad": vad_list,
            "words": text_list,
        }
