from pathlib import Path
from omegaconf import OmegaConf
from natsort import natsorted

from .util import select_platform_path, repo_root

# Register once at import time
if not OmegaConf.has_resolver("select_platform_path"):
    OmegaConf.register_new_resolver("select_platform_path", select_platform_path)

def dirname_if_file(path: str) -> str:
    p = Path(path)
    return str(p.parent) if p.is_file() or p.suffix else str(p)

if not OmegaConf.has_resolver("dirname"):
    OmegaConf.register_new_resolver("dirname", dirname_if_file)

def checkpoint_path(path: str, pretrained: bool) -> str:
    p = Path(path)
    result = str(p)

    if pretrained and not p.suffix:
        candidate = p / "last.ckpt"
        if candidate.exists():
            result = str(candidate)
        else:
            parent = p.parent
            if parent.exists():
                folders = natsorted([d for d in parent.iterdir() if d.is_dir()], reverse=True)
                if folders:
                    if p in folders:
                        idx = folders.index(p)
                    else:
                        idx = min(range(len(folders)), key=lambda i: abs(i - len(folders)//2))
                    nearest = folders[idx]
                    candidate = nearest / "last.ckpt"
                    if candidate.exists():
                        result = str(candidate)
                    else:
                        result = str(p / "last.ckpt")
            else:
                result = str(p / "last.ckpt")

    return result

if not OmegaConf.has_resolver("checkpoint_path"):
    OmegaConf.register_new_resolver("checkpoint_path", checkpoint_path)

def hydra_run_dir(save: bool, log_base_dir: str) -> str:
    result = "."
    if save:
        result = log_base_dir
    return result

if not OmegaConf.has_resolver("hydra_run_dir"):
    OmegaConf.register_new_resolver("hydra_run_dir", hydra_run_dir)

def repo_root_resolver(_: str = None) -> str:
    """Hydra resolver wrapper for util.repo_root()."""
    return repo_root()

if not OmegaConf.has_resolver("repo_root"):
    OmegaConf.register_new_resolver("repo_root", repo_root_resolver)

# Re-export so you can import it from here
__all__ = ["OmegaConf"]
