# -*- coding: utf-8 -*-
import sys
import threading
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

base_path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(base_path))
sys.path = list(dict.fromkeys(sys.path))

import numpy as np
from tqdm import tqdm
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from src.libs.audio_io.audio_reader import CAudioReader
from src.libs.data.audio_data.audio_format import CAudioFormat
from src.libs.logger.log import getLogger, load_logger_config
from src.libs.speech_recognition.VAD.vad_listener_interface import IVADListener
from src.libs.speech_recognition.VAD.vad_manager import CVADManager
from src.libs.utils import OmegaConf


AUDIO_EXTENSIONS = {
    ".aac", ".aiff", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma"
}


class CVADExtractor(IVADListener):
    """Extract VAD labels through the VVAD reader/manager pipeline.

    Args:
        cfg_dict: root Hydra configuration.
        src_paths: audio files to process.
        overwrite: replace existing output files.
        timeout: maximum processing time per file in seconds.
        output_name: name of the output file written beside each source audio.
    """

    def __init__(self, cfg_dict: DictConfig, src_paths: List[Path], overwrite: bool = False,
                 timeout: float = 300.0, output_name: str = "vad_extracted.txt"):
        self._logger = getLogger(self.__class__.__name__)
        self.cfg_dict = cfg_dict
        self.src_paths = [Path(path) for path in src_paths]
        self.overwrite = overwrite
        self.timeout = timeout
        self.output_name = output_name
        self._labels = {}
        self._finished = threading.Event()
        self._error = None

    def extract_vad_from_audio(self, i: int) -> Optional[Path]:
        src_path = self.src_paths[i]
        tgt_path = src_path.parent / self.output_name
        if tgt_path.exists() and not self.overwrite:
            self._logger.info(f"[{i}] {tgt_path} already exists, skipping extraction")
            return tgt_path

        self._logger.info(f"[{i}] Processing {src_path} -> {tgt_path}")
        self._labels = {}
        self._error = None
        self._finished.clear()

        audio_config = self.cfg_dict.io.audio_io.audio_format
        audio_format = CAudioFormat(
            sample_rate=audio_config.sample_rate,
            number_channels=audio_config.num_channels,
            bit_depth=audio_config.format,
        )
        buffer_length = self.cfg_dict.preprocessing.audio_processing.buffer_length
        chunk_size = int(audio_format.sample_rate * buffer_length)

        vad_manager = CVADManager(self.cfg_dict, mode="AudioChunks", type="vad")
        audio_reader = CAudioReader(
            source=str(src_path),
            threaded=False,
            chunk_size=chunk_size,
            format=audio_format,
            show_signal=False,
            resample_per=100,
            save_audio=False,
            save_interval=None,
            audio_path=None,
        )
        vad_manager.add_listener(self)
        audio_reader.add_listener(vad_manager)

        try:
            vad_manager.start()
            audio_reader.start()
            start_time = time.monotonic()
            while not self._finished.wait(timeout=0.1):
                if self._error is not None:
                    raise RuntimeError(f"VAD failed for {src_path}") from self._error
                if time.monotonic() - start_time > self.timeout:
                    raise TimeoutError(f"VAD extraction timed out for {src_path}")
        finally:
            audio_reader.stop()
            vad_manager.stop()

        if self._error is not None:
            raise RuntimeError(f"VAD failed for {src_path}") from self._error

        rows = []
        for index in sorted(self._labels):
            time_init = round(buffer_length * index, 3)
            time_end = round(buffer_length * (index + 1), 3)
            rows.append(f"[{time_init}, {time_end}]: {int(self._labels[index])}")
        np.savetxt(tgt_path, np.asarray(rows, dtype=object), fmt="%s", delimiter=",")
        return tgt_path

    def extract(self) -> List[Path]:
        extracted = []
        for i in tqdm(range(len(self.src_paths)), desc="Extracting VAD"):
            result = self.extract_vad_from_audio(i)
            if result is not None:
                extracted.append(result)
        return extracted

    def on_vad(self, index_first_speech: int):
        self._logger.debug(f"Speech starts at chunk {index_first_speech}")

    def on_sd(self, index_last_speech: int):
        self._logger.debug(f"Speech ends at chunk {index_last_speech}")

    def on_processed_chunk(
            self, vad_results_indexes: List[Tuple[int, bool]], is_last_one: bool):
        self._labels.update(dict(vad_results_indexes))
        if is_last_one:
            self._finished.set()

    def on_end_vad(self):
        self._finished.set()

    def on_error_vad(self):
        self._error = RuntimeError("VAD manager reported an error")
        self._finished.set()


def find_audio_files(paths: Iterable[Path], recursive: bool = True) -> List[Path]:
    audio_files = []
    for path in paths:
        path = path.resolve()
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            audio_files.append(path)
        elif path.is_dir():
            iterator = path.rglob("*") if recursive else path.glob("*")
            audio_files.extend(
                candidate for candidate in iterator
                if candidate.is_file() and candidate.suffix.lower() in AUDIO_EXTENSIONS
            )
    return sorted(set(audio_files))


def _validate_output_paths(src_paths: List[Path], output_name: str):
    output_paths = [path.parent / output_name for path in src_paths]
    if len(output_paths) != len(set(output_paths)):
        raise ValueError(
            f"Multiple audio files share a folder and would write the same {output_name}")


@hydra.main(config_path="../src/configs", config_name="config", version_base=None)
def main(cfg_dict: DictConfig):
    load_logger_config(cfg_dict.logger)
    logger = getLogger("Main")

    OmegaConf.resolve(cfg_dict)

    extractor_cfg = cfg_dict.recognition.vad_extractor
    paths = [Path(to_absolute_path(str(path))) for path in extractor_cfg.paths]
    src_paths = find_audio_files(paths, recursive=extractor_cfg.recursive)
    if not src_paths:
        raise FileNotFoundError(f"No supported audio files found in {paths}")
    logger.info(f"Found {len(src_paths)} audio files")
    _validate_output_paths(src_paths, extractor_cfg.output_name)

    CVADExtractor(cfg_dict, src_paths, overwrite=extractor_cfg.overwrite,
                  timeout=extractor_cfg.timeout, output_name=extractor_cfg.output_name).extract()


if __name__ == "__main__":
    main()
