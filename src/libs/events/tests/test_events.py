# coding: UTF-8
import os
import sys
from pathlib import Path

base_path = Path(__file__).resolve().parent.parent.parent.parent.parent.resolve()

sys.path.insert(0, str(base_path / 'src' / 'libs'))
sys.path.insert(0, str(base_path / 'src'))
sys.path.insert(0, str(base_path / 'src' / 'libs' / 'logger'))
sys.path.insert(0, str(base_path / 'src' / 'libs' / 'utils'))
sys.path.insert(0, str(base_path / 'src' / 'libs' / 'events'))
sys.path = list(dict.fromkeys(sys.path))

import argparse
import torch
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt

from objective import ObjectiveVAP
from events import TurnTakingEvents, EventConfig
from prepare_data.multimodal_datamodule import CMultimodalDataModule
from configs.configuration import CBaseConfig
from logger import load_logger_config, getLogger, set_logger_level
from utils import (get_run_name, recursive_clean, OmegaConf, ensure_dataset_files,
                   plot_mel_spectrogram, plot_vad, plot_waveform, plot_event, plot_stereo_mel_spec,
                   plot_next_speaker_probs, log_mel_spectrogram)


parser = argparse.ArgumentParser(description='EVENTS TEST')
parser.add_argument('-log_level', '--log_level', type=str, default="",
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'DEV', 'TRACE'],
                    help='Logging level override (optional)')
args, remaining_argv = parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining_argv


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg_dict: DictConfig) -> None:
    OmegaConf.resolve(cfg_dict)

    load_logger_config(cfg_dict.logger)
    main_logger = getLogger("Main")

    if len(args.log_level) != 0:
        set_logger_level("Main", args.log_level)

    main_logger.info("Events test started")

    # ------------------------------------------------------------
    # DATA PREPARATION
    # ------------------------------------------------------------
    src_dir = Path(cfg_dict.dataset.path)
    db_path = Path(cfg_dict.dataset.extracted_path)

    keyword = cfg_dict.dataset.dataset_selected.keyword
    chunk_duration = cfg_dict.data.audio_duration
    horizon = cfg_dict.data.vad_horizon
    stride = cfg_dict.data.audio_overlap
    base_name = f"audio_duration_{chunk_duration}_horizon_{horizon}_stride_{stride}_"
    name = get_run_name(cfg_dict)

    src_train_path = db_path / (base_name + f'train_{keyword.lower()}.csv')
    src_valid_path = db_path / (base_name + f'valid_{keyword.lower()}.csv')
    src_test_path = db_path / (base_name + f'test_{keyword.lower()}.csv')
    ensure_dataset_files(src_train_path, src_valid_path, src_test_path)

    clean_cfg_dict = OmegaConf.to_container(cfg_dict, resolve=True, throw_on_missing=True)
    base_conf = CBaseConfig(cfg_dict)

    main_logger.dev("Creating data module")
    kwargs = {
        "train_path": src_train_path,
        "val_path": src_valid_path,
        "test_path": src_test_path,
        "horizon": horizon,
        "frame_hz": cfg_dict.model.encoder.frame_hz,
        "sample_rate": cfg_dict.data.sample_rate,
        "mono": cfg_dict.data.audio_mono,
        "batch_size": 1,
        "shuffle": cfg_dict.train.training_features.shuffle,
        "num_workers": cfg_dict.train.training_features.num_workers,
        "frame_hz": cfg_dict.model.encoder.frame_hz,
        "multimodal": cfg_dict.data.multimodal,
        "use_face_encoder": cfg_dict.train.training_features.use_face_encoder or
                            cfg_dict.train.training_features.use_backbone,
        "use_cache": cfg_dict.data.use_cache,
        "exclude_av_cache": cfg_dict.data.exclude_av_cache,
        "preload_av": cfg_dict.data.preload_av,
        "cache_dir": os.path.abspath(f"{cfg_dict.dataset.path}/tmp_cache2"),
        "use_multiprocess": cfg_dict.train.training_features.use_multiprocess,
        "manager": None,
        "lock": None
        }
    dm = CMultimodalDataModule(**kwargs)

    conf = EventConfig(
        metric_time=0.05,
        equal_hold_shift=False,
        sh_pre_cond_time=0.5,
        sh_post_cond_time=0.5,
        )
    eventer = TurnTakingEvents(conf)
    ob = ObjectiveVAP()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for batch in dm.val_dataloader():
        events = eventer(batch["vad"][:, :-100])
        ds_labels = ob.get_labels(batch["vad"].to(device)).cpu()
        for b in range(2):
            x = torch.arange(batch["vad"].shape[1] - 100) / 50
            plt.close("all")
            fig, ax = plt.subplots(3, 1, sharex=True, figsize=(12, 4))

            plot_mel_spectrogram(y=batch["waveform"][:, b], ax=ax)
            #_ = plot_waveform(waveform=batch["waveform"][0, b], ax=ax[b], label=f"Speaker{b}")

            plot_vad(x, batch["vad"][b, :-100, 0], ax=ax[0], ypad=5)
            plot_vad(x, batch["vad"][b, :-100, 1], ax=ax[1], ypad=5)
            plot_event(events["shift"][b], ax=ax, color="g")
            plot_event(events["hold"][b], ax=ax, color="b")
            plot_event(events["short"][b], ax=ax)
            ax[-1].plot(x, ds_labels[b], linewidth=2)
            ax[-1].set_ylim([0, 2])
            # ax[c].axvline(s/50, color='g', linewidth=2)
            # ax[c].axvline(e/50, color='r', linewidth=2)
            plt.tight_layout()
            plt.show()
            # plt.pause(0.1)

        # Forward (stereo->mono)
        waveform = batch["waveform"].mean(-2, keepdim=True)
        mel_spec = log_mel_spectrogram(waveform, hop_length=320)

        ###################################################
        # Figure
        ###################################################
        #fig, ax = plt.subplots(3, 1, figsize=(12, 8))
        #_, ax_mels = plot_stereo_mel_spec(
        #    waveform, mel_spec=mel_spec, vad=batch["vad"], ax=[ax[0], ax[2]], plot=False
        #)
        #ax[1] = plot_next_speaker_probs(
            #p_ns=probs["p"][0],
            #ax=ax[1],
            #p_bc=probs["bc_prediction"][0],
            #vad=d["vad"][0],
            #alpha_ns=0.8,
            #legend=True,
        #)
        plt.show()


if __name__ == "__main__":
    main()
