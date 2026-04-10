import sys
import os
from pathlib import Path

base_path = Path(__file__).resolve().parent
sys.path.insert(0, str(base_path))
sys.path.insert(0, str((base_path.parent / 'logger').resolve()))
sys.path = list(dict.fromkeys(sys.path))

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from os.path import exists
from os import cpu_count
from typing import Optional
import csv
from pathlib import Path
from multiprocessing import Manager, Lock

from .multimodal_dataset import CMultimodalTurnTakingDataset
from src.libs.logger.log import getLogger


class CMultimodalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_path: Optional[str] = "",
        val_path: Optional[str] = "",
        test_path: Optional[str] = "",
        horizon: float = 2,
        frame_hz: int = 50,
        sample_rate: int = 16000,
        flip_channels: bool = True,
        flip_probability: float = 0.5,
        mask_vad: bool = True,
        mask_vad_probability: float = 0.4,
        mono: bool = False,
        batch_size: int = 4,
        shuffle: bool = False,
        num_workers: int = 2,
        pin_memory: bool = True,
        multimodal: bool = False,
        use_face_encoder: bool = False,
        use_cache: bool = False,
        exclude_av_cache: bool = False,
        preload_av: bool = False,
        cache_dir: str = 'tmp_cache',
        use_multiprocess: bool = False,
        manager: Manager = None,
        lock:Lock = None,
    ):
        super().__init__()

        self._logger = getLogger(self.__class__.__name__)
        # Files
        self._train_loader = None
        self._val_loader = None
        self._test_loader = None

        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path

        # values
        self.horizon = horizon
        self.sample_rate = sample_rate
        self.frame_hz = frame_hz

        # Transforms
        self.mono = mono
        self.flip_channels = flip_channels
        self.flip_probability = flip_probability
        self.mask_vad = mask_vad
        self.mask_vad_probability = mask_vad_probability

        # DataLoder
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        
        self.multimodal = multimodal
        self.use_face_encoder = use_face_encoder
        
        self.use_cache = use_cache
        self.exclude_av_cache = exclude_av_cache
        self.preload_av = preload_av
        self.cache_dir = cache_dir
        
        self.use_multiprocess = use_multiprocess
        self.manager = manager
        self.lock = lock

        self._dataset_kwargs = {
            "use_multiprocess": self.use_multiprocess,
            "horizon": self.horizon,
            "sample_rate": self.sample_rate,
            "frame_hz": self.frame_hz,
            "mono": self.mono,
            "num_pool": os.cpu_count() // 2,
            "use_cache": self.use_cache,
            "multimodal": self.multimodal,
            "use_face_encoder": self.use_face_encoder,
            "exclude_av_cache": self.exclude_av_cache,
            "preload_av": self.preload_av,
            "cache_dir": self.cache_dir,
        }

        self._loader_kwargs = {
            "batch_size": self.batch_size,
            "pin_memory": self.pin_memory,
            "num_workers": self.num_workers,
            "shuffle": False,
            "persistent_workers": self.num_workers > 0,
            "drop_last": True,
        }

        if self.num_workers > 0:
            self._loader_kwargs["prefetch_factor"] = int(self.batch_size / 2)

        self.load_data()

    def load_data(self) -> None:
        if self.train_path is not None and str(self.train_path).strip():
            assert Path(self.train_path).exists(), f"No TRAIN file found: {self.train_path}"

        if self.val_path is not None and str(self.val_path).strip():
            assert Path(self.val_path).exists(), f"No VAL file found: {self.val_path}"

        if self.test_path is not None and str(self.test_path).strip():
            assert Path(self.test_path).exists(), f"No TEST file found: {self.test_path}"

        if self.train_path is not None:
            self.train_dset = CMultimodalTurnTakingDataset(
                self.train_path,
                manager=self.manager,
                lock=self.lock,
                split="Train",
                shuffle=self.shuffle,
                config_dict=self._dataset_kwargs
            )

        if self.val_path is not None:
            self.val_dset = CMultimodalTurnTakingDataset(
                self.val_path,
                manager=self.manager,
                lock=self.lock,
                split="Val",
                shuffle=False,
                config_dict=self._dataset_kwargs
            )
            
        if self.test_path is not None:
            self.test_dset = CMultimodalTurnTakingDataset(
                self.test_path,
                manager=self.manager,
                lock=self.lock,
                split="Test",
                shuffle=False,
                config_dict=self._dataset_kwargs
            )
        
        self.cache_lookup = {}
        if self.use_cache:
            raise NotImplementedError("Cache lookup is not implemented yet")

            cache_dir = Path(self.cache_dir)

            self.cache_csv = cache_dir / 'cache_multimodal_{}.csv'.format(Path(self.train_path).stem)
            self.cache_lookup = self.load_cache(self.cache_lookup, self.cache_csv)

            self.cache_csv = cache_dir / 'cache_multimodal_{}.csv'.format(Path(self.val_path).stem)
            self.cache_lookup = self.load_cache(self.cache_lookup, self.cache_csv)

            self.cache_csv = cache_dir / 'cache_multimodal_{}.csv'.format(Path(self.test_path).stem)
            self.cache_lookup = self.load_cache(self.cache_lookup, self.cache_csv)
        
        if self.preload_av:
            audio_dict, image_dict = self.train_dset.preload(cache_lookup=self.cache_lookup)
            _, _ = self.val_dset.preload(audio_dict=audio_dict, image_dict=image_dict)
            _, _ = self.test_dset.preload(audio_dict=audio_dict, image_dict=image_dict)

    def setup(self, stage: Optional[str] = "fit"):
        """Loads the datasets"""

        if stage in (None, "fit"):
            pass

        if stage in (None, "test"):
            pass

    def train_dataloader(self):
        if self._train_loader is None:
            self._train_loader = DataLoader(self.train_dset, **self._loader_kwargs)
        return self._train_loader

    def val_dataloader(self):
        if self._val_loader is None:
            self._val_loader = DataLoader(self.val_dset, **self._loader_kwargs)
        return self._val_loader

    def test_dataloader(self):
        if self._test_loader is None:
            self._test_loader = DataLoader(self.test_dset, **self._loader_kwargs)
        return self._test_loader
    
    def load_cache(self, cache_lookup, cache_csv):
        with open(cache_csv, 'r') as f:
            reader = csv.reader(f)
            for data in reader:
                cache_lookup[(data[0], data[1], data[2])] = data[3]

        return cache_lookup

    def __repr__(self):
        s = self.__class__.__name__
        s += f"\n\tTrain: {self.train_path}"
        s += f"\n\tVal: {self.val_path}"
        s += f"\n\tTest: {self.test_path}"
        s += f"\n\tHorizon: {self.horizon}"
        s += f"\n\tSample rate: {self.sample_rate}"
        s += f"\n\tFrame Hz: {self.frame_hz}"
        s += f"\nData"
        s += f"\n\tbatch_size: {self.batch_size}"
        s += f"\n\tpin_memory: {self.pin_memory}"
        s += f"\n\tnum_workers: {self.num_workers}"
        s += f"\nTransform"
        s += f"\n\tflip_channels: {self.flip_channels}"
        s += f"\n\tflip_probability: {self.flip_probability}"
        s += f"\n\tmask_vad: {self.mask_vad}"
        s += f"\n\tmask_vad_probability: {self.mask_vad_probability}"
        return s
