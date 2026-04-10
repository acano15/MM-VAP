# coding: UTF-8
import os
import sys
from pathlib import Path

base_path = Path(__file__).resolve().parent
sys.path.insert(0, str(base_path))
sys.path.insert(0, str(base_path.parent / 'libs'))
sys.path = list(dict.fromkeys(sys.path))

import random
from typing import Optional, Dict
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
from tqdm import tqdm

from dialog_audio_dm_config import CDialogueAudioDMConfig
from database_selection import CDataBaseSelection
from log import getLogger


class CDialogAudioDM(pl.LightningDataModule):
    """
        Dialogue Audio DataModule.

        Args:
            cfg_dict (OmegaConf): Configuration dictionary with all parameters required
                to set up the audio data module, including preprocessing, dataloader,
                and label configuration.
            manager (Optional[object]): External process or resource manager for dataset
                coordination or logging, if applicable.
            lock (Optional[object]): Optional synchronization lock, useful for multiprocessing
                or multithreaded data access.
        """
    def __init__(self, cfg_dict: OmegaConf,
                 manager: Optional[object] = None,
                 lock: Optional[object] = None):
        super().__init__()
        self._config = CDialogueAudioDMConfig(cfg_dict)
        self._manager = manager
        self._lock = lock

        self._logger = getLogger(self.__class__.__name__)

    def prepare_data(self):
        """
        loads the data over all splits.
        Using huggingface datasets which may process/download and cache the data or used cache versions.

        Doing this here to make sure that all the data is accessable before any training, evaluation etc.
        However, uses the same call as is done in `self.setup()`

        So this might be unnecessary given we use Huggingface `datasets` ...

        To avoid the `datasets` logging warnings set `DATASETS_VERBOSITY=error` in your terminal ENV.
        """
        for ds_name in self._config.datasets:
            self._logger.info(f"Preparing dataset {ds_name}")
            database_selected = CDataBaseSelection().get_db_by_name(ds_name)
            dataset = database_selected(a_cfg=cfg, a_manager=manager, a_lock=lock)

        for split in ["train", "validation", "test"]:
            _ = get_dialog_audio_datasets(
                datasets=self.datasets,
                split=split,
            )

    def _dataset(self, dset, split="train"):
        # Only flip during training...
        if split == "train":
            flip = self.flip_channels
            undersampling = self.undersampling
            oversampling = self.oversampling
        elif split == "val":
            flip = False
            undersampling = False
            oversampling = False
        elif split == "test":
            flip = False
            undersampling = False
            oversampling = False
        else:
            print("SPLIT ERROR")
            exit(1)

        return DialogAudioDataset(
            dataset=dset,
            transforms=self.transforms,
            type=self.type,
            audio_mono=self.audio_mono,
            audio_duration=self.audio_duration,
            audio_overlap=self.audio_overlap,
            audio_normalize=self.audio_normalize,
            sample_rate=self.sample_rate,
            vad=self.vad,
            vad_hz=self.vad_hz,
            vad_horizon=self.vad_horizon,
            vad_history=self.vad_history,
            vad_history_times=self.vad_history_times,
            flip_channels=flip,
            flip_probability=0.5,
            label_type=self.label_type,
            bin_times=self.bin_times,
            pre_frames=self.pre_frames,
            threshold_ratio=self.threshold_ratio,
            undersampling=undersampling,
            oversampling=oversampling
        )

    def setup(self, stage: Optional[str] = "fit"):
        """Loads the datasets"""
        
        # Define a helper function to avoid repetitive code
        def get_dataset(split: str):
            return get_dialog_audio_datasets(
                datasets=self.datasets,
                split=split,
                train_files=self.train_files,
                val_files=self.val_files,
                test_files=self.test_files
            )
        
        if stage == "fit":
            self.train_dset = self._dataset(get_dataset("train"), split="train")
            self.val_dset = self._dataset(get_dataset("val"), split="val")
            self.test_dset = None

        elif stage == "test":
            self.train_dset = None
            self.val_dset = None
            self.test_dset = self._dataset(get_dataset("test"), split="test")

        else:
            self.train_dset = self._dataset(get_dataset("train"), split="train")
            self.val_dset = self._dataset(get_dataset("val"), split="val")
            self.test_dset = self._dataset(get_dataset("test"), split="test")
    
    def _pad_tensor(self, tensor, target_size, dim=-1):
        if tensor.size(dim) < target_size:
            padding_size = target_size - tensor.size(dim)
            padding = torch.zeros(
                *tensor.size()[:dim], padding_size, *tensor.size()[dim + 1 :]
            )
            tensor = torch.cat((tensor, padding), dim)
        elif tensor.size(dim) > target_size:
            tensor = tensor.narrow(dim, 0, target_size)
        return tensor


    def collate_fn(self, batch):
        ret = {key: [] for key in self.keys}

        for b in batch:
            for key in self.keys:
                if key in b:
                    ret[key].append(
                        # self._pad_tensor(b[key], self.target_sizes[key], self.dimensions.get(key, -1))
                        b[key]
                    )

        ret = {key: torch.cat(value) for key, value in ret.items() if len(value) > 0}
        ret["dset_name"] = [b["dataset_name"] for b in batch]
        ret["session"] = [b["session"] for b in batch]

        return ret

    def get_full_sample(self, split="val"):
        if split == "train":
            return self.train_dset.get_full_sample()
        elif split == "val":
            return self.val_dset.get_full_sample()
        elif split == "test":
            return self.test_dset.get_full_sample()
        else:
            return None

    def change_frame_mode(self, mode="False"):
        if self.train_dset is not None:
            self.train_dset.change_frame_mode(mode)
        if self.val_dset is not None:
            self.val_dset.change_frame_mode(mode)
        if self.test_dset is not None:
            self.test_dset.change_frame_mode(mode)

    def seed_worker(self, worker_id):
        np.random.seed(worker_id)
        random.seed(worker_id)

    def train_dataloader(self):
        return DataLoader(
            self.train_dset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            shuffle=True,
            worker_init_fn=self.seed_worker,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            shuffle=False,
            worker_init_fn=self.seed_worker,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            shuffle=False,
            worker_init_fn=self.seed_worker,
        )

    def __repr__(self):
        s = "DialogAudioDM"
        s += f"\n\tbatch_size: {self.batch_size}"
        s += f"\n\tpin_memory: {self.pin_memory}"
        s += f"\n\tnum_workers: {self.num_workers}"

        if hasattr(self, "train_dset"):
            s += "\n\t" + ("-" * 10) + "\n"
            s += str(self.train_dset)
        elif hasattr(self, "test_dset"):
            s += "\n\t" + ("-" * 10) + "\n"
            s += str(self.train_dset)
        return s

    @staticmethod
    def print_dm(data_conf, args=None):
        print("-" * 60)
        print("Dataloader")
        for k, v in data_conf["dataset"].items():
            print(f"  {k}: {v}")
        if args is not None:
            print("  batch_size: ", args.batch_size)
            print("  num_workers: ", args.num_workers)
        print()

    @staticmethod
    def default_config_path():
        return DEFAULT_CONFIG

    @staticmethod
    def load_config(path=None, args=None, format="dict") -> Dict:
        if path is None:
            path = DialogAudioDM.default_config_path()
        return load_config(path, args=args, format=format)

    @staticmethod
    def add_data_specific_args(parent_parser):
        """argparse arguments for SoSIModel (based on yaml-config)"""
        parser = parent_parser.add_argument_group("ULMProjection")
        parser.add_argument("--data_conf", default=None, type=str)
        parser.add_argument("--batch_size", default=4, type=int)
        parser.add_argument("--num_workers", default=cpu_count(), type=int)
        parser.add_argument("--train_files", default=None, type=str)
        parser.add_argument("--val_files", default=None, type=str)
        parser.add_argument("--test_files", default=None, type=str)

        # A workaround for OmegaConf + WandB-Sweeps
        conf = DialogAudioDM.load_config()
        parser = OmegaConfArgs.add_argparse_args(parser, conf)
        return parent_parser


if __name__ == "__main__":
    data_conf = DialogAudioDM.load_config()

    dm = DialogAudioDM(
        datasets="noxi",
        audio_duration=10,
        audio_overlap=9.5,
        vad_hz=25,
        num_workers=0,
    )

    dm.setup(None)
    print(dm)

    print("\nBATCH DATASET")
    d = dm.val_dset[0]
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {tuple(v.shape)}")
        else:
            print(f"{k}: {v}")

    print("FULL SAMPLE")
    d = dm.get_full_sample("val")
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {tuple(v.shape)}")
        else:
            print(f"{k}: {v}")

    for i in range(len(dm.val_dset)):
        d = dm.val_dset[i]

    print("DATALOADER TEST")
    pbar_val = tqdm(
        enumerate(dm.train_dataloader()),
        total=len(dm.train_dataloader()),
    )
    for ii, batch in pbar_val:
        pass
    pbar_val = tqdm(
        enumerate(dm.val_dataloader()),
        total=len(dm.val_dataloader()),
    )
    for ii, batch in pbar_val:
        pass

    print("Frame Mode ON")
    dm.change_frame_mode(True)
    pbar_val = tqdm(
        enumerate(dm.val_dataloader()),
        total=len(dm.val_dataloader()),
    )
    for ii, batch in pbar_val:
        pass
    pbar_val = tqdm(
        enumerate(dm.test_dataloader()),
        total=len(dm.test_dataloader()),
    )
    for ii, batch in pbar_val:
        pass
