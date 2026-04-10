import os
import pandas as pd
import numpy as np
import torch
import torchaudio
import json
from sklearn.utils import shuffle as sk_shuffle
from pathlib import Path
from typing import Optional, Dict, Any
from multiprocessing import Manager, Lock
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import Dataset
from typing import Tuple, Dict, Any, Set

from .multimodal_dataset_config import CConfigMultimodalTurnTakingDataset
from src.libs.utils.audio import load_waveform
from src.libs.utils.util import vad_list_to_onehot, vad_list_to_onehot_windowed
from src.libs.logger.log import getLogger


class CMultimodalTurnTakingDataset(Dataset):
    """
    Multimodal dataset class for turn-taking modeling. Loads data using a CSV metadata file,
    and applies a configuration dictionary via `CConfigMultimodalTurnTakingDataset`.

    Each sample includes:
    - Audio waveform (clipped to time window)
    - VAD (voice activity detection) labels
    - Transcription (if provided)
    - Visual features: gaze, head pose, face landmarks, body joints
    - Optional: preprocessed face image tensors (if use_face_encoder=True)

    Args:
        file_path (str or Path): Path to the metadata .csv file
        manager (multiprocessing.Manager, optional): Shared multiprocessing manager
        lock (multiprocessing.Lock, optional): Shared multiprocessing lock
        split (str): Dataset split name (e.g., "train", "test")
        shuffle (bool): Whether to shuffle the dataset
        config_dict (dict): Dictionary matching `CConfigMultimodalTurnTakingDataset.REQUIRED_KEYS`
    """

    def __init__(
        self,
        file_path: str,
        manager: Optional[Manager] = None,
        lock: Optional[Lock] = None,
        split: str = "train",
        shuffle: bool = False,
        config_dict: Dict[str, Any] = None
        ):
        super().__init__()

        self._config = CConfigMultimodalTurnTakingDataset(config_dict)

        self._logger = getLogger(self.__class__.__name__)

        self.file_path = Path(file_path)
        self.manager = manager
        self.lock = lock
        self.split = split
        self.shuffle = shuffle

        self._cache_memory = {}
        self._keep_ids = set()
        self._prefetch_future = None
        self._prefetch_id = None
        self._executor = ThreadPoolExecutor(max_workers=1)

        self.df = self._load_df()
        if len(self.df) == 0:
            self._logger.error(f"Loaded empty dataset from {self.file_path}")
            raise ValueError(f"Loaded empty dataset from {self.file_path}")

        self._logger.info(f"Initialized dataset with {len(self.df)} samples from {self.file_path}")
        if self._config.use_cache:
            self._logger.warning("Not implemented yet use_cache=True")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = {}

        row = self.df.iloc[idx]
        data["dataset"] = row['dataset']
        data["session"] = row['session']

        file_id = row['id']
        self._keep_ids = {str(file_id)}
        if self._prefetch_id is not None:
            self._keep_ids.add(self._prefetch_id)

        self._cleanup_not_used_files(self._keep_ids)

        start_time = row['start']
        end_time = row['end']
        duration = end_time - start_time

        # Load audio
        audio_path = str(row["audio_path"]).replace('\\', '/')
        audio_name = os.path.basename(audio_path)
        self._logger.trace(f"Loading audio file {audio_name}")
        if "mix" in audio_name:
            mono = False
        else:
            mono = self._config.mono

        if self._check_in_cache_memory(audio_path):
            audio = self._cache_memory[audio_path]
        else:
            audio, sr = load_waveform(audio_path, sample_rate=self._config.sample_rate, mono=mono)
            if sr != self._config.sample_rate:
                self._logger.warning(f"Sample rate mismatch for {audio_name}: "
                                     f"expected {self._config.sample_rate}, got {sr}. Resampling.")
                audio = torchaudio.transforms.Resample(sr, self._config.sample_rate)(audio)
            self._cache_memory[audio_path] = audio

        start_sample = int(start_time * self._config.sample_rate)
        end_sample = int(end_time * self._config.sample_rate)
        waveform = audio[:, start_sample:end_sample]
        data["waveform"] = waveform.contiguous()
        self._logger.dev(f"Loaded waveform with shape {waveform.shape} for {audio_name}")

        # Load VAD
        vad = vad_list_to_onehot_windowed(row["vad_list"], duration=duration + self._config.horizon,
                                          start_time=start_time, end_time=end_time,
                                          frame_hz=self._config.frame_hz)
        data["vad"] = vad.contiguous()
        self._logger.dev(f"Loaded VAD with shape {vad.shape} for {audio_name}")

        ref_start_ratio, ref_end_ratio = self._get_ref_ratio(audio, start_time, end_time,
                                                             self._config.sample_rate)
        tgt_size = round(self._config.frame_hz * duration)

        if self._config.multimodal:
            # Precomputed body features
            for key in ["gaze", "head", "face", "body"]:
                for i in ["1", "2"]:
                    path = row[f"{key}_path{i}"].replace("\\", "/")
                    self._logger.trace(f"Loading {key} feature from speaker {i} file {os.path.basename(path)}")
                    try:
                        if not self._check_in_cache_memory(path):
                            feature = self._load_csv_with_filter(path)
                            self._cache_memory[path] = feature
                        else:
                            feature = self._cache_memory[path]
                        feature_chunk = self._get_chunk(
                            feature, ref_start_ratio, ref_end_ratio, tgt_size)
                        data[f"{key}{i}"] = feature_chunk.contiguous()
                        self._logger.dev(f"Loaded feature {key}{i} with shape {feature_chunk.shape}")
                    except Exception as e:
                        self._logger.error(f"Error loading {key}{i} at path: {path} -> {e}")
                        raise e

        if self._config.use_face_encoder:
            # Load face images
            for i in ["1", "2"]:
                face_path = row[f"face_im_path{i}"].replace("\\", "/")
                self._logger.trace(f"Loading face image from speaker {i} file {os.path.basename(face_path)}")
                try:
                    if not self._check_in_cache_memory(face_path):
                        tensor = self._load_array_to_tensor(face_path)
                        self._cache_memory[face_path] = tensor
                    else:
                        tensor = self._cache_memory[face_path]
                    face_chunk = self._get_chunk(tensor, ref_start_ratio, ref_end_ratio, tgt_size)
                    data[f"face_im{i}"] = face_chunk.contiguous()
                    self._logger.dev(f"Loaded face_im{i} with shape {face_chunk.shape}")
                except Exception as e:
                    self._logger.error(f"Error loading face_im{i} at path: {face_path} -> {e}")
                    raise e

        self._schedule_prefetch(idx + 1)
        return data

    def _load_df(self):
        if not self.file_path.exists():
            self._logger.error(f"Metadata CSV not found at {self.file_path}")
            raise FileNotFoundError(f"Metadata CSV not found at {self.file_path}")

        def _vl(x):
            return json.loads(x)

        def _session(x):
            return str(x)

        converters = {
            "vad_list": _vl,
            "session": _session,
            }
        df_result = pd.read_csv(self.file_path, converters=converters)
        self._logger.debug(f"Loaded metadata CSV file {self.file_path} with {len(df_result)} rows")
        if self.shuffle:
            self._logger.dev("Shuffling dataset")
            grouped = list(df_result.groupby('id'))
            shuffled_within_groups = [group.sample(frac=1, random_state=42) for _, group in grouped]
            shuffled_groups = sk_shuffle(shuffled_within_groups, random_state=42)
            df_result = pd.concat(shuffled_groups, ignore_index=True)

        return df_result

    def _load_csv_with_filter(self, csv_path):
        csv_path = csv_path.replace('\\', '/')

        data = torch.from_numpy(
            pd.read_csv(
                csv_path
                ).filter(regex="^(?!.*confidence).*$", axis=1).values
            ).type(torch.get_default_dtype())

        return data

    def _load_array_to_tensor(self, src_path):
        src_path = src_path.replace('\\', '/')

        src_path = Path(src_path)

        if src_path.suffix == '.npy':
            # data = np.load(src_path)
            data = np.load(src_path, mmap_mode='r')
            src_tensor = torch.from_numpy(data)
        elif src_path.suffix == '.pt':
            src_tensor = torch.load(src_path)
        else:
            assert False, '{} should be .npy or .pt'.format(str(src_path))

        return src_tensor

    def _cleanup_not_used_files(self, keep_ids: Set[str]) -> None:
        """Remove cached entries whose basename doesn't contain any keep id."""
        keys_to_remove = []
        for key in list(self._cache_memory.keys()):
            base = os.path.basename(key)
            if not any(kid in base for kid in keep_ids):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            self._logger.dev(f"Removing {key} from cache memory")
            del self._cache_memory[key]

    def _check_in_cache_memory(self, file_path: str) -> bool:
        """
        Check if the file is already in cache memory

        Args:
            file_path: path to the file

        Returns:
            bool: True if file is in cache memory, False otherwise
        """
        result = False
        if file_path in self._cache_memory.keys():
            result = True
            self._logger.dev(f"File {file_path} found in cache memory")
        else:
            self._logger.dev(f"File {file_path} not found in cache memory")

        return result

    @staticmethod
    def _get_chunk(src, ref_start_ratio, ref_end_ratio, tgt_size):
        start_index = round(src.size(0) * ref_start_ratio)
        end_index = round(src.size(0) * ref_end_ratio)

        if (end_index - start_index) < tgt_size:
            tgt = src[-tgt_size:]
        elif (end_index - start_index) > tgt_size:
            tgt = src[end_index - tgt_size:end_index]
        else:
            tgt = src[start_index:end_index]

        if tgt.size(0) != tgt_size:
            tgt = src[-tgt_size:]

        return tgt

    @staticmethod
    def _get_ref_ratio(data, ref_start_sec, ref_end_sec, ref_sample_rate):
        ref_start_ratio = (ref_start_sec * ref_sample_rate) / data.shape[1]
        ref_end_ratio = (ref_end_sec * ref_sample_rate) / data.shape[1]
        return ref_start_ratio, ref_end_ratio

    def _schedule_prefetch(self, next_idx: int) -> None:
        if next_idx >= len(self.df):
            return

        next_row = self.df.iloc[next_idx]
        next_id = str(next_row["id"])

        # Already prefetched / in-flight
        if self._prefetch_id == next_id and self._prefetch_future is not None:
            return

        # If previous prefetch finished, surface exceptions early
        if self._prefetch_future is not None and self._prefetch_future.done():
            _ = self._prefetch_future.result()  # raises if failed
            self._prefetch_future = None

        self._prefetch_id = next_id
        self._prefetch_future = self._executor.submit(self._prefetch_id_into_cache, next_row)

    def _prefetch_id_into_cache(self, row: pd.Series) -> None:
        """Load all files for this row into _cache_memory (CPU), best-effort."""
        file_id = str(row["id"])

        # Keep both current (unknown here) + this id from being cleaned meanwhile
        self._keep_ids.add(file_id)

        # Audio
        audio_path = str(row["audio_path"]).replace("\\", "/")
        if audio_path not in self._cache_memory:
            audio_name = os.path.basename(audio_path)
            mono = False if "mix" in audio_name else self._config.mono
            audio, sr = load_waveform(audio_path, sample_rate=self._config.sample_rate, mono=mono)
            if sr != self._config.sample_rate:
                audio = torchaudio.transforms.Resample(sr, self._config.sample_rate)(audio)
            self._cache_memory[audio_path] = audio

        if self._config.multimodal:
            for key in ["gaze", "head", "face", "body"]:
                for i in ["1", "2"]:
                    path = str(row[f"{key}_path{i}"]).replace("\\", "/")
                    if path not in self._cache_memory:
                        self._cache_memory[path] = self._load_csv_with_filter(path)

        if self._config.use_face_encoder:
            for i in ["1", "2"]:
                face_path = str(row[f"face_im_path{i}"]).replace("\\", "/")
                if face_path not in self._cache_memory:
                    self._cache_memory[face_path] = self._load_array_to_tensor(face_path)
