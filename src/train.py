# coding: UTF-8
import os
import sys
from pathlib import Path

base_path = Path(__file__).resolve().parent
sys.path.insert(0, str(base_path.parent / 'src' / 'libs'))
sys.path = list(dict.fromkeys(sys.path))

import warnings
from multiprocessing import Manager
import hydra
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (RichProgressBar, ModelCheckpoint, EarlyStopping,
                                         LearningRateMonitor)
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.tuner.tuning import Tuner

from prepare_data.datamodule import VapDataModule
from model.model import VAPModel
from model.callbacks import (CManualStopCallback, SymmetricSpeakersCallback,
                             AudioAugmentationCallback, ResetEpochCallback,
                             OverrideEpochStepCallback)
from configs.configuration import CBaseConfig
from utils import get_run_name, recursive_clean, repo_root, select_platform_path
from logger import load_logger_config, getLogger


# everything_deterministic()
warnings.simplefilter("ignore")
OmegaConf.register_new_resolver("select_platform_path", select_platform_path)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg_dict: DictConfig) -> None:
    OmegaConf.resolve(cfg_dict)

    load_logger_config(cfg_dict.logger)

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

    src_train_path = db_path / (base_name + f'train_{keyword.lower()}.csv')
    src_valid_path = db_path / (base_name + f'valid_{keyword.lower()}.csv')
    src_test_path = db_path / (base_name + f'test_{keyword.lower()}.csv')

    clean_cfg_dict = OmegaConf.to_container(cfg_dict, resolve=True, throw_on_missing=True)
    manager = Manager()
    lock = manager.Lock()

    if cfg_dict.general.use_seed:
        pl.seed_everything(15)

    base_conf = CBaseConfig(cfg_dict)
    global_batch_size = cfg_dict.train.training_features.global_batch_size
    batch_size = cfg_dict.train.training_features.batch_size
    assert global_batch_size % batch_size == 0, "(global_batch_size / batch_size) should be 0"
    grad_accum = int(global_batch_size / batch_size)

    limit_train_batches = cfg_dict.train.training_features.limit_train_batches
    limit_val_batches = cfg_dict.train.training_features.limit_val_batches
    limit_test_batches = cfg_dict.train.training_features.limit_test_batches

    dm = VapDataModule(
        train_path=src_train_path,
        val_path=src_valid_path,
        test_path=src_test_path,
        horizon=2,
        batch_size=batch_size,
        num_workers=cfg_dict.train.training_features.num_workers,
        frame_hz=cfg_dict.model.encoder.frame_hz,
        multimodal=cfg_dict.data.multimodal,
        use_face_encoder=cfg_dict.data.use_face_encoder,
        use_cache=False,
        exclude_av_cache=False,
        preload_av=False,
        cache_dir=os.path.abspath(f"{cfg_dict.dataset.path}/tmp_cache4"),
        manager=manager,
        lock=lock,
        )

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------
    torch.set_float32_matmul_precision("medium")
    model = VAPModel(
        conf=base_conf,
        hyperparameters=cfg_dict.train.hyperparameters,
        lr_scheduler_config=cfg_dict.train.callbacks.lr_scheduler
        )
    model.load_pretrained_parameters(
        base_conf.pretrained_vap,
        base_conf.pretrained_cpc,
        base_conf.pretrained_face_encoder)

    logger = TensorBoardLogger(
        save_dir=cfg_dict.logger.log_dir,
        name=None,
        default_hp_metric=False
        )
    logger.save = lambda *args, **kwargs: None
    safe_cfg_dict = recursive_clean(clean_cfg_dict)
    logger.hparams = {}
    logger.log_hyperparams(safe_cfg_dict)

    name = get_run_name(cfg_dict)

    # ------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------
    callbacks = [
        RichProgressBar(),
        CManualStopCallback(),
        SymmetricSpeakersCallback(),
        OverrideEpochStepCallback(),
        LearningRateMonitor(logging_interval='epoch')
        ]
    if not cfg_dict.train.training_features.save_checkpoints:
        callbacks.append(ResetEpochCallback())

    callbacks.append(
        EarlyStopping(
            monitor=cfg_dict.train.callbacks.early_stopping.monitor,
            mode=cfg_dict.train.callbacks.params.mode,
            patience=cfg_dict.train.callbacks.params.patience,
            strict=True,  # crash if "monitor" is not found in val metrics
            verbose=True
            )
        )
    if cfg_dict.train.training_features.save_checkpoints:
        callbacks.append(
            ModelCheckpoint(
                dirpath=cfg_dict.train.training_features.model_checkpoints_path,
                monitor=cfg_dict.train.callbacks.checkpoint.monitor,
                mode=cfg_dict.train.callbacks.checkpoint.mode,
                save_top_k=cfg_dict.train.callbacks.checkpoint.save_top_k,
                save_last=cfg_dict.train.callbacks.checkpoint.save_last,
                every_n_epochs=cfg_dict.train.callbacks.checkpoint.every_n_epochs,
                auto_insert_metric_name=cfg_dict.train.callbacks.checkpoint.auto_insert_metric_name,
                filename=f"{name}-{cfg_dict.train.callbacks.checkpoint.filename}"
                )
            )

    if cfg_dict.train.hyperparameters.find_learning_rate:
        lr_finder_callbacks = [cb for cb in callbacks if not isinstance(cb, ModelCheckpoint)]
        trainer = pl.Trainer(
            max_epochs=cfg_dict.train.training_features.max_epochs,
            logger=None,
            callbacks=lr_finder_callbacks,
            strategy="auto",
            accumulate_grad_batches=grad_accum,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            limit_test_batches=limit_test_batches,
            check_val_every_n_epoch=cfg_dict.train.training_features.val_execution_freq,
            log_every_n_steps=min(50, limit_train_batches, limit_val_batches, limit_test_batches),
            enable_progress_bar=True,
            enable_model_summary=True,
            enable_checkpointing=False,
            deterministic=cfg_dict.general.use_seed,
            use_distributed_sampler=True)

        tuner = Tuner(trainer)

        # finds learning rate automatically
        lr_finder = tuner.lr_find(model, datamodule=dm)
        fig = lr_finder.plot(suggest=True)
        fig.show()
        model.lr = lr_finder.suggestion()
        print(f"Learning rate suggestion is {lr_finder.suggestion()}")

    # ------------------------------------------------------------
    # Training and validation data loaders
    # ------------------------------------------------------------

    trainer = pl.Trainer(
        max_epochs=cfg_dict.train.training_features.max_epochs,
        logger=logger,
        callbacks=callbacks,
        strategy="auto",
        accumulate_grad_batches=grad_accum,
        # fast_dev_run=3,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        log_every_n_steps=min(50, limit_train_batches, limit_val_batches, limit_test_batches),
        # profiler="simple",
        # profiler="advanced",
        # profiler="pytorch",
        # precision=16,
        # progress_bar_refresh_rate=0,
        enable_progress_bar=True,
        enable_model_summary=True,
        enable_checkpointing=cfg_dict.train.training_features.save_checkpoints,
        deterministic=cfg_dict.general.use_seed,
        benchmark=False, # Enables cuDNN auto-tuner for fastest convolution algorithms
        use_distributed_sampler=True,
        detect_anomaly=True,
        num_sanity_val_steps=0)

    if cfg_dict.train.training_features.pretrained:
        ckpt_path = cfg_dict.train.training_features.model_checkpoints_path
    else:
        ckpt_path = None
    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
    trainer.validate(model, datamodule=dm)

    # ------------------------------------------------------------
    # Testing model
    # ------------------------------------------------------------
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()
