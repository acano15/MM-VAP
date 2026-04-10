import torch
from torch import Tensor
import torchaudio
import torchaudio.functional as AF
import torchaudio.transforms as AT
from typing import Any, Dict, Optional, Tuple

from functools import partial

from .util import find_island_idx_len

print = partial(print, flush=True)

SAMPLE_RATE = 16_000
N_MELS = 80
N_FFT = 400
HOP_LENGTH = 320
TRIAD_HOLD: Tensor = torch.tensor([[0, 1, 0], [3, 1, 3]])  # on silence


def time_to_samples(t: float, sample_rate: int) -> int:
    return int(t * sample_rate)


def time_to_frames(t: float, hop_time: float) -> int:
    return int(t / hop_time)


def sample_to_time(n_samples: int, sample_rate: int) -> float:
    return n_samples / sample_rate


def get_audio_info(audio_path: str) -> Dict[str, Any]:
    info = torchaudio.info(audio_path)
    return {
        "name": audio_path,
        "duration": sample_to_time(info.num_frames, info.sample_rate),
        "sample_rate": info.sample_rate,
        "num_frames": info.num_frames,
        "bits_per_sample": info.bits_per_sample,
        "num_channels": info.bits_per_sample,
        "encoding": info.encoding,
    }


def load_waveform(
    path: str,
    sample_rate: Optional[int] = 16000,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    mono: bool = False,
) -> Tuple[torch.Tensor, int]:

    # print('Loading:', path, ' - start')

    if start_time is None and end_time is None:
        x, sr = torchaudio.load(path)
    else:
        info = get_audio_info(path)

        start_frame = 0
        if start_time is not None:
            start_frame = time_to_samples(start_time, info["sample_rate"])

        end_frame = info["num_frames"]
        if end_time is not None:
            end_frame = time_to_samples(end_time, info["sample_rate"])

        num_frames = end_frame - start_frame
        x, sr = torchaudio.load(path, frame_offset=start_frame, num_frames=num_frames)

    if mono and x.shape[0] > 1:
        x = x.mean(dim=0).unsqueeze(0)

    if sample_rate is not None:
        if sr != sample_rate:
            x = AF.resample(x, orig_freq=sr, new_freq=sample_rate)
            sr = sample_rate

    # print('Loading:', path, ' - done')

    return x, sr


def log_mel_spectrogram(
    waveform: torch.Tensor,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    sample_rate: int = SAMPLE_RATE,
) -> torch.Tensor:
    mel_spec = AT.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        normalized=True,
    )(waveform)
    log_mel_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_mel_spec = torch.maximum(log_mel_spec, log_mel_spec.max() - 8.0)
    log_mel_spec = (log_mel_spec + 4.0) / 4.0
    return log_mel_spec


def get_dialog_states(vad: Tensor) -> Tensor:
    """Vad to the full state of a 2 person vad dialog
    0: only speaker 0
    1: none
    2: both
    3: only speaker 1
    """
    assert vad.ndim >= 1
    return (2 * vad[..., 1] - vad[..., 0]).long() + 1


def fill_pauses(
    vad: Tensor,
    ds: Tensor,
    islands: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
    ) -> Tensor:
    assert vad.ndim == 2, "fill_pauses require ds=(n_frames, 2)"
    assert ds.ndim == 1, "fill_pauses require ds=(n_frames,)"

    filled_vad = vad.clone()

    if islands is None:
        s, d, v = find_island_idx_len(ds)
    else:
        s, d, v = islands

    # less than three entries means that there are no pauses
    # requires at least: activity-from-speaker  ->  Silence   --> activity-from-speaker
    if len(v) < 3:
        return vad

    triads = v.unfold(0, size=3, step=1)
    next_speaker, steps = torch.where(
        (triads == TRIAD_HOLD.unsqueeze(1).to(triads.device)).sum(-1) == 3
        )
    for ns, pre in zip(next_speaker, steps):
        cur = pre + 1
        # Fill the matching template
        filled_vad[s[cur]: s[cur] + d[cur], ns] = 1.0
    return filled_vad
