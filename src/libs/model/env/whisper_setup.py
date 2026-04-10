# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import types

base_path = Path(__file__).resolve().parent
sys.path.insert(0, str((base_path.parent.parent / 'logger').resolve()))
sys.path.insert(0, str((base_path.parent.parent / 'utils').resolve()))
sys.path = list(dict.fromkeys(sys.path))

from util import repo_root, load_module_from_file, load_package_from_dir
whisper_repo_path = (Path(repo_root()) / 'external' / 'whisper-flamingo').resolve()
whisper_pkg = load_package_from_dir(whisper_repo_path / "whisper", alias="whisper_flamingo_whisper")
sys.modules["whisper"] = whisper_pkg
wf_utils = load_module_from_file(whisper_repo_path / "utils.py", alias="whisper_flamingo_utils")
sys.modules["whisper_flamingo_utils"] = wf_utils

import numpy as np
import hashlib
import io
import urllib
import warnings
from typing import List, Optional, Union
import torch
from tqdm import tqdm

from whisper_flamingo_whisper import (
    Whisper,
    ModelDimensions,
    _download,
    _MODELS,
    _ALIGNMENT_HEADS,
    available_models,
)


def setup_whisper_flamingo_environment() -> None:
    """Full environment setup for Whisper-Flamingo + AV-HuBERT + Fairseq.

    Includes:
        - path registration
        - numpy / omegaconf patches
        - dynamic Whisper-Flamingo import
        - AV-HuBERT Fairseq registration
    """
    add_whisper_flamingo_folders()
    patch_numpy_types()
    patch_omegaconf()
    original_utils = sys.modules.get("utils", None)
    if original_utils is not None:
        sys.modules.pop("utils")  # let hubert_asr resolve its own local utils.py
    import importlib
    importlib.invalidate_caches()
    import whisper_flamingo_whisper as _wfw
    global _original_load_model
    _original_load_model = _wfw.load_model  # capture original
    _wfw.load_model = _patched_load_model  # install patch

    try:
        import hubert_asr  # noqa: F401
    except:
        import avhubert.hubert_asr
    if original_utils is not None:
        sys.modules["utils"] = original_utils  # restore your project's utils


def add_whisper_flamingo_folders() -> None:
    """Ensure all critical submodules are discoverable by Python."""
    repo_root_path = Path(repo_root()).resolve().absolute()
    paths = [
        repo_root_path / "external" / "av_hubert" / "avhubert",
        repo_root_path / "external" / "av_hubert" / "fairseq",
        repo_root_path / "external" / "av_hubert",
        repo_root_path / "external" / "whisper-flamingo",
    ]
    for p in reversed(paths):
        sp = str(p.resolve())
        if sp not in sys.path:
            sys.path.insert(0, sp)
    sys.path = list(dict.fromkeys(sys.path))  # dedup


def patch_numpy_types() -> None:
    """Ensure backward compatibility with np.float/np.int/np.bool."""
    for t in [("float", float), ("int", int), ("bool", bool)]:
        if not hasattr(np, t[0]):
            setattr(np, t[0], t[1])


def patch_omegaconf() -> None:
    """Fix omegaconf._utils.is_primitive_type missing in newer versions."""
    import omegaconf._utils as oc_utils
    if not hasattr(oc_utils, "is_primitive_type"):
        import numbers
        def is_primitive_type(t):
            return t in (str, bool, bytes) or issubclass(t, numbers.Number)
        oc_utils.is_primitive_type = is_primitive_type

def load_model(
    name: str,
    device: Optional[Union[str, torch.device]] = None,
    download_root: str = None,
    in_memory: bool = False,
    dropout_rate: float = 0.0,
    video: bool = False,
    video_model_path: str = "",
    av_hubert_path: str = "av_hubert/avhubert",
    prob_av: float = 0.0,
    prob_a: float = 0.0,
    av_hubert_encoder: bool = False,
    av_fusion: str = "early",
    add_adapter: bool = False,
    adapter_dim: int = 256,
    add_gated_x_attn: int = 0,
) -> Whisper:
    """
    Load a Whisper ASR model

    Parameters
    ----------
    name : str
        one of the official model names listed by `whisper.available_models()`, or
        path to a model checkpoint containing the model dimensions and the model state_dict.
    device : Union[str, torch.device]
        the PyTorch device to put the model into
    download_root: str
        path to download the model files; by default, it uses "~/.cache/whisper"
    in_memory: bool
        whether to preload the model weights into host memory

    Returns
    -------
    model : Whisper
        The Whisper ASR model instance
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if download_root is None:
        default = os.path.join(os.path.expanduser("~"), ".cache")
        download_root = os.path.join(os.getenv("XDG_CACHE_HOME", default), "whisper")

    if name in _MODELS:
        checkpoint_file = _download(_MODELS[name], download_root, in_memory)
        alignment_heads = _ALIGNMENT_HEADS[name]
    elif os.path.isfile(name):
        checkpoint_file = open(name, "rb").read() if in_memory else name
        alignment_heads = None
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {available_models()}"
        )

    with (
        io.BytesIO(checkpoint_file) if in_memory else open(checkpoint_file, "rb")
    ) as fp:
        checkpoint = torch.load(fp, map_location=device)
    del checkpoint_file

    dims = ModelDimensions(**checkpoint["dims"])
    model = Whisper(dims, dropout_rate, video, video_model_path, av_hubert_path, prob_av, prob_a,
                    av_hubert_encoder, av_fusion, add_adapter, adapter_dim, add_gated_x_attn)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    # if alignment_heads is not None:
    #     model.set_alignment_heads(alignment_heads)

    return model.to(device)

def _patched_load_model(*args, **kwargs):
    from fairseq.data import dictionary
    from fairseq.data.encoders.sentencepiece_bpe import SentencepieceBPE
    from sentencepiece import SentencePieceProcessor
    import torch as _torch
    import builtins
    import logging
    import warnings
    _old_print = builtins.print
    _old_warn = warnings.warn
    _old_log_level = logging.root.manager.disable

    builtins.print = lambda *a, **kw: None
    warnings.warn = lambda *a, **kw: None
    logging.disable(logging.CRITICAL)

    _orig_load = _torch.load  # save original

    def _safe_torch_load(*a, **kw):
        _torch.serialization.add_safe_globals([
            dictionary.Dictionary,
            SentencepieceBPE,
            SentencePieceProcessor
        ])
        return _orig_load(*a, **kw)  # call original, not patched

    _torch.load = _safe_torch_load
    result = _original_load_model(*args, **kwargs)

    builtins.print = _old_print
    warnings.warn = _old_warn
    logging.disable(_old_log_level)
    return result
