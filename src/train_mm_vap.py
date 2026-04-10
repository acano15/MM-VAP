# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

base_path = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(base_path))
sys.path = list(dict.fromkeys(sys.path))

import warnings
from multiprocessing import Manager
import json
import tempfile
import hydra
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.tuner.tuning import Tuner

from src.libs.prepare_data.datamodule import VapDataModule
from src.libs.prepare_data.multimodal_datamodule import CMultimodalDataModule
from src.libs.model.model import VAPModel
from src.libs.model.vap_module import CVAPModule
from src.libs.model.callbacks import (CEarlyStoppingCallback, CMetricPlotCallback, CCustomProgressBar,
                             CManualStopCallback, SymmetricSpeakersCallback,
                             AudioAugmentationCallback, ResetEpochCallback,
                             OverrideEpochStepCallback)
from src.configs.configuration import CBaseConfig
from src.libs.utils import get_run_name, recursive_clean, OmegaConf, ensure_dataset_files
from src.libs.logger.log import load_logger_config, getLogger


# everything_deterministic()
warnings.simplefilter("ignore")


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg_dict: DictConfig) -> None:

    OmegaConf.resolve(cfg_dict)

    load_logger_config(cfg_dict.logger)
    main_logger = getLogger("Main")
    main_logger.record(f"Configuration file: \n "
                  f"{json.dumps(OmegaConf.to_container(cfg_dict, resolve=True), indent=4)} \n")

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
    src_train_path = db_path / (base_name + f'train{keyword.lower()}.csv')
    src_valid_path = db_path / (base_name + f'valid{keyword.lower()}.csv')
    src_test_path = db_path / (base_name + f'test{keyword.lower()}.csv')
    ensure_dataset_files(cfg_dict.dataset.selected_dataset, src_train_path, src_valid_path,
                         src_test_path)

    clean_cfg_dict = OmegaConf.to_container(cfg_dict, resolve=True, throw_on_missing=True)
    manager = Manager()
    lock = manager.Lock()

    if cfg_dict.general.use_seed:
        pl.seed_everything(15)
        cfg_dict.train.training_features.use_deterministic = "warn"

    base_conf = CBaseConfig(cfg_dict)
    global_batch_size = cfg_dict.train.training_features.global_batch_size
    batch_size = cfg_dict.train.training_features.batch_size
    assert global_batch_size % batch_size == 0, "(global_batch_size / batch_size) should be 0"
    grad_accum = int(global_batch_size / batch_size)

    limit_train_batches = cfg_dict.train.training_features.limit_train_batches
    limit_val_batches = cfg_dict.train.training_features.limit_val_batches
    limit_test_batches = cfg_dict.train.training_features.limit_test_batches

    main_logger.dev("Creating data module")
    kwargs = {
        "train_path": src_train_path,
        "val_path": src_valid_path,
        "test_path": src_test_path,
        "horizon": horizon,
        "frame_hz": cfg_dict.model.encoder.frame_hz,
        "sample_rate": cfg_dict.data.sample_rate,
        "mono": cfg_dict.data.audio_mono,
        "batch_size": batch_size,
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
        "manager": manager,
        "lock": lock
        }
    if cfg_dict.general.use_baseline:
        main_logger.info("Using baseline VapDataModule")
        dm = VapDataModule(**kwargs)
    else:
        dm = CMultimodalDataModule(**kwargs)

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------
    main_logger.dev("Creating model")
    torch.set_float32_matmul_precision("medium")

    if cfg_dict.general.use_baseline:
        model = VAPModel(conf=base_conf,
                         hyperparameters=cfg_dict.train.hyperparameters,
                         lr_scheduler_config=cfg_dict.train.callbacks.lr_scheduler)
    else:
        if cfg_dict.train.training_features.pretrained:
            ckpt_path = cfg_dict.train.training_features.model_checkpoints_path
            model = CVAPModule.load_from_checkpoint(
                ckpt_path, conf=cfg_dict.model.configuration,
                hyperparameters=cfg_dict.train.hyperparameters,
                lr_scheduler_config=cfg_dict.train.callbacks.lr_scheduler)
            checkpoint = torch.load(ckpt_path, weights_only=False)
            max_epoch = cfg_dict.train.training_features.max_epochs
            current_epoch = int(checkpoint.get("epoch"))
            cfg_dict.train.training_features.max_epochs = max(max_epoch, current_epoch)
            cfg_dict.train.training_features.initial_epoch = current_epoch
            main_logger.info("Using pretrained model, loading pretrained parameters from "
                             f"checkpoints path: {ckpt_path}. Starting from epoch {current_epoch}")
        else:
            model = CVAPModule(
                conf=cfg_dict.model.configuration,
                hyperparameters=cfg_dict.train.hyperparameters,
                lr_scheduler_config=cfg_dict.train.callbacks.lr_scheduler)

    model.load_pretrained_parameters(
        base_conf.pretrained_vap,
        base_conf.pretrained_cpc,
        base_conf.pretrained_face_encoder)
    if cfg_dict.train.training_features.summarize.use:
        model.summarize(max_depth=cfg_dict.train.training_features.summarize.depth)

    # ------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------

    main_logger.info("Creating callbacks")
    checkpoints_path = Path(cfg_dict.train.training_features.model_checkpoints_path)

    if cfg_dict.train.training_features.pretrained and checkpoints_path.suffix:
        cfg_dict.train.training_features.model_checkpoints_path = str(
            checkpoints_path.parent.resolve())
        main_logger.info(f"Model checkpoints will be loaded from: {checkpoints_path}")
    else:
        checkpoints_path = None

    callbacks = [
        CCustomProgressBar(),
        CManualStopCallback(),
        SymmetricSpeakersCallback(),
        OverrideEpochStepCallback(),
        LearningRateMonitor(logging_interval='epoch')
        ]
    if cfg_dict.train.training_features.save_behavior_images:
        metric_plots_dir = os.path.join(cfg_dict.train.training_features.model_checkpoints_path,
                                        "images")
        main_logger.info(f"Metric images will be saved to: {metric_plots_dir}")
        callbacks.append(CMetricPlotCallback(save_dir=metric_plots_dir, log_to_tensorboard=True))

    if not cfg_dict.train.training_features.save_checkpoints:
        callbacks.append(ResetEpochCallback())

    callbacks.append(
        CEarlyStoppingCallback(
            modality=cfg_dict.train.callbacks.early_stopping.modality,
            monitor=cfg_dict.train.callbacks.early_stopping.monitor,
            mode=cfg_dict.train.callbacks.params.mode,
            patience=cfg_dict.train.callbacks.params.patience,
            strict=True,  # crash if "monitor" is not found in val metrics
            verbose=True
            )
        )

    if cfg_dict.train.training_features.save_checkpoints:
        main_logger.info("Model checkpoints will be saved to: "
                         f"{cfg_dict.train.training_features.model_checkpoints_path}")
        ckpt_cb = ModelCheckpoint(
            dirpath=cfg_dict.train.training_features.model_checkpoints_path,
            monitor=cfg_dict.train.callbacks.checkpoint.monitor,
            mode=cfg_dict.train.callbacks.checkpoint.mode,
            save_top_k=cfg_dict.train.callbacks.checkpoint.save_top_k,
            save_last=cfg_dict.train.callbacks.checkpoint.save_last,
            every_n_epochs=cfg_dict.train.callbacks.checkpoint.every_n_epochs,
            auto_insert_metric_name=cfg_dict.train.callbacks.checkpoint.auto_insert_metric_name,
            filename=f"{name}-{cfg_dict.train.callbacks.checkpoint.filename}")
        callbacks.append(ckpt_cb)

    tb_save_dir = cfg_dict.logger.log_dir
    if not cfg_dict.general.save:
        tb_save_dir = tempfile.mkdtemp()
    else:
        main_logger.info(f"Tensorboard destination folder is {tb_save_dir}")

    logger = TensorBoardLogger(
        save_dir=tb_save_dir,
        name=None,
        default_hp_metric=False
        )
    logger.save = lambda *args, **kwargs: None
    safe_cfg_dict = recursive_clean(clean_cfg_dict)
    logger.hparams = {}
    logger.log_hyperparams(safe_cfg_dict, {"z_dummy": 0.0})

    # ------------------------------------------------------------
    # Training and validation data loaders
    # ------------------------------------------------------------

    main_logger.info("Initializing training and validation data loaders")
    if cfg_dict.train.hyperparameters.find_learning_rate:
        lr_finder_callbacks = [cb for cb in callbacks if not isinstance(cb, ModelCheckpoint)]
        main_logger.debug("Initializing lr auto finder")
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
            profiler="advanced",
            enable_progress_bar=True,
            enable_model_summary=False,
            enable_checkpointing=False,
            deterministic=cfg_dict.train.training_features.use_deterministic)

        tuner = Tuner(trainer)

        # finds learning rate automatically
        lr_finder = tuner.lr_find(model, datamodule=dm)
        fig = lr_finder.plot(suggest=True)
        fig.show()
        model.lr = lr_finder.suggestion()
        main_logger.info(f"Learning rate suggestion is {lr_finder.suggestion()}")

    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        max_epochs=cfg_dict.train.training_features.max_epochs,
        logger=logger,
        callbacks=callbacks,
        strategy="auto",
        accumulate_grad_batches=grad_accum,
        # fast_dev_run=3,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        log_every_n_steps=50,
        profiler="simple", #simple, advancced, pytorch,
        #precision="16-mixed",
        # progress_bar_refresh_rate=0,
        reload_dataloaders_every_n_epochs=False,
        enable_progress_bar=True,
        enable_model_summary=cfg_dict.train.training_features.summarize.use,
        enable_checkpointing=cfg_dict.train.training_features.save_checkpoints,
        deterministic=cfg_dict.train.training_features.use_deterministic,
        benchmark=True, # Enables cuDNN auto-tuner for fastest convolution algorithms
        use_distributed_sampler=True,
        detect_anomaly=False,
        num_sanity_val_steps=0)

    main_logger.dev("Initializing training")

    trainer.fit(model, datamodule=dm, ckpt_path=checkpoints_path)
    trainer.validate(model, datamodule=dm)
    main_logger.info("Training finished")

    # ------------------------------------------------------------
    # Testing model
    # ------------------------------------------------------------
    main_logger.info("Performing test with the last model checkpoint")
    trainer.test(model, datamodule=dm, ckpt_path=None)
    if cfg_dict.train.training_features.save_checkpoints:
        best_model_path = ckpt_cb.best_model_path
        main_logger.info(f"Performing test with the best model checkpoint {best_model_path}")
        trainer.test(model=None, datamodule=dm, ckpt_path=best_model_path)


if __name__ == "__main__":
    main()
