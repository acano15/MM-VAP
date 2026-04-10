# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

base_path = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(base_path))
sys.path = list(dict.fromkeys(sys.path))

import warnings
import hydra
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig

from src.libs.prepare_data.multimodal_datamodule import CMultimodalDataModule
from src.libs.model.vap_module import CVAPModule
from src.libs.model.callbacks import (CCustomProgressBar, CManualStopCallback,
                                          SymmetricSpeakersCallback, OverrideEpochStepCallback)
from src.libs.utils import get_run_name, OmegaConf
from src.libs.logger.log import load_logger_config, getLogger


# everything_deterministic()
warnings.simplefilter("ignore")


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg_dict: DictConfig) -> None:
    OmegaConf.resolve(cfg_dict)

    load_logger_config(cfg_dict.logger)
    main_logger = getLogger("Main")

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

    if keyword is not None:
        keyword = f"_{keyword}"
    else:
        keyword = ""

    src_test_path = db_path / (base_name + f'test{keyword.lower()}.csv')
    clean_cfg_dict = OmegaConf.to_container(cfg_dict, resolve=True, throw_on_missing=True)
    main_logger.dev("Creating data module")
    kwargs = {
        "train_path": None,
        "val_path": None,
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

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------
    main_logger.dev("Creating model")
    torch.set_float32_matmul_precision("medium")

    ckpt_path = cfg_dict.train.training_features.model_checkpoints_path
    model = CVAPModule.load_from_checkpoint(
        ckpt_path, conf=cfg_dict.model.configuration,
        hyperparameters=cfg_dict.train.hyperparameters,
        lr_scheduler_config=cfg_dict.train.callbacks.lr_scheduler).eval()
    checkpoint = torch.load(ckpt_path, weights_only=False)
    max_epoch = cfg_dict.train.training_features.max_epochs
    current_epoch = int(checkpoint.get("epoch"))
    cfg_dict.train.training_features.max_epochs = max(max_epoch, current_epoch)
    cfg_dict.train.training_features.initial_epoch = current_epoch
    main_logger.info("Using pretrained model, loading pretrained parameters from checkpoints path: "
                     f"{ckpt_path}. Starting from epoch {current_epoch}")

    # ------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------

    main_logger.info("Creating callbacks")
    checkpoints_path = Path(cfg_dict.train.training_features.model_checkpoints_path)

    cfg_dict.train.training_features.model_checkpoints_path = str(
        checkpoints_path.parent.resolve())
    main_logger.info(f"Model checkpoints will be loaded from: {checkpoints_path}")

    callbacks = [
        CCustomProgressBar(),
        CManualStopCallback(),
        SymmetricSpeakersCallback(),
        OverrideEpochStepCallback(),
        ]

    # ------------------------------------------------------------
    # TEST
    # ------------------------------------------------------------
    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        logger=None,
        callbacks=callbacks,
        limit_test_batches=cfg_dict.train.training_features.limit_test_batches,
        enable_progress_bar=True,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=cfg_dict.train.training_features.use_deterministic,
        benchmark=True,
        num_sanity_val_steps=0,
        )

    main_logger.info(f"Running test with checkpoint: {ckpt_path}")
    results = trainer.test(model=model, datamodule=dm, ckpt_path=ckpt_path)

    main_logger.info(f"Test results: {results}")


if __name__ == "__main__":
    main()
