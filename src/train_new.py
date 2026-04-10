# coding: UTF-8
import os
import sys
from pathlib import Path

base_path = Path(__file__).resolve().parent
sys.path.insert(0, str(base_path.parent / 'src' / 'libs'))
sys.path.insert(0, str(base_path.parent / 'src'))
sys.path.insert(0, str(base_path.parent / 'src' / 'libs' / 'logger'))
sys.path.insert(0, str(base_path.parent / 'src' / 'libs' / 'utils'))
sys.path = list(dict.fromkeys(sys.path))

import warnings
from multiprocessing import Manager
from typing import Dict, Any, List
import json
import tempfile
import hydra
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, open_dict
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.callbacks import Callback
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import traceback

from prepare_data.datamodule import VapDataModule
from prepare_data.multimodal_datamodule import CMultimodalDataModule
from model.model import VAPModel
from model.vap_module import CVAPModule
from model.callbacks import (CEarlyStoppingCallback, CMetricPlotCallback, CCustomProgressBar,
                             CManualStopCallback, SymmetricSpeakersCallback,
                             AudioAugmentationCallback, ResetEpochCallback,
                             OverrideEpochStepCallback)
from configs.configuration import CBaseConfig
from utils import get_run_name, recursive_clean, repo_root, OmegaConf, ensure_dataset_files
from logger import load_logger_config, getLogger


# everything_deterministic()
warnings.simplefilter("ignore")


class PatchedPruningCallback(PyTorchLightningPruningCallback, Callback):
    """Fix for 'Expected a parent' error with PyTorch Lightning >= 1.8."""
    pass


def objective(logger, trial, cfg_dict: DictConfig, kwargs: Dict[str, Any], callbacks: List) -> (float):
    try:
        cfg_copy = OmegaConf.create(OmegaConf.to_container(cfg_dict, resolve=False))
        for class_name, class_conf in cfg_copy.logger.classes.items():
            if class_name != logger.name:
                class_conf.level = "WARNING"

        load_logger_config(cfg_copy.logger)
        base_lr = cfg_copy.train.hyperparameters.learning_rate
        cfg_copy.train.hyperparameters.learning_rate = trial.suggest_float(
            "learning_rate", base_lr / 100, base_lr * 100)

        # -----------------------------
        # Model architecture hyperparameters
        # -----------------------------
        cfg_copy.model.encoder.output_layer = trial.suggest_int(
            "channel_layers", 1, 12, step=2)
        cfg_copy.model.encoder.cross_layers = trial.suggest_int(
            "cross_layers", 3, 12, step=3)
        cfg_copy.model.model_kwargs.Transformer.num_heads = trial.suggest_categorical(
            "num_heads", [2, 4, 8, 16, 32])
        cfg_copy.model.audio_module.dropout = trial.suggest_float(
            "dropout", 0.3, 0.8)

        # -----------------------------
        # Training options hyperparameters
        # -----------------------------
        cfg_copy.train.training_features.global_batch_size = trial.suggest_categorical(
            "global_batch_size", [8, 64, 128, 256])
        cfg_copy.train.training_features.batch_size = trial.suggest_categorical(
            "batch_size", [4, 8, 16])
        cfg_copy.train.hyperparameters.optimizer = trial.suggest_categorical(
            "optimizer", ["Adam", "AdamW", "SGD"])
        cfg_copy.train.hyperparameters.learning_rate = trial.suggest_float(
            "learning_rate", 1e-5, 1e-2)

        global_batch_size = cfg_copy.train.training_features.global_batch_size
        batch_size = cfg_copy.train.training_features.batch_size
        if global_batch_size % batch_size != 0:
            candidates = [h for h in range(1, global_batch_size + 1) if global_batch_size % h == 0]
            if not candidates:
                raise ValueError(
                    f"No valid batch_size divisor found for global_batch_size={global_batch_size}")
            batch_size = min(candidates, key=lambda h: abs(h - batch_size))
            cfg_copy.train.training_features.batch_size = batch_size
        grad_accum = int(global_batch_size / batch_size)

        # -----------------------------
        # Data module
        # -----------------------------
        dm = VapDataModule(**kwargs) if cfg_copy.general.use_baseline else CMultimodalDataModule(**kwargs)

        # -----------------------------
        # Model
        # -----------------------------
        base_conf = CBaseConfig(cfg_copy)
        model = (
            VAPModel(
                conf=base_conf,
                hyperparameters=cfg_copy.train.hyperparameters,
                lr_scheduler_config=cfg_copy.train.callbacks.lr_scheduler,
            )
            if cfg_copy.general.use_baseline
            else CVAPModule(
                conf=cfg_dict.model.configuration,
                hyperparameters=cfg_copy.train.hyperparameters,
                lr_scheduler_config=cfg_copy.train.callbacks.lr_scheduler,
            )
        )

        model.load_pretrained_parameters(
            base_conf.pretrained_vap,
            base_conf.pretrained_cpc,
            base_conf.pretrained_face_encoder,
        )

        limit_train_batches = (
            cfg_copy.train.training_features.limit_train_batches * cfg_copy.train.hparams_search.percentage_usage
        )
        limit_val_batches = (
            cfg_copy.train.training_features.limit_val_batches * cfg_copy.train.hparams_search.percentage_usage
        )

        # -----------------------------
        # Trainer
        # -----------------------------
        logger_tb = TensorBoardLogger(
            save_dir=cfg_copy.tb_save_dir,
            name=f"trial_{trial.number}",
            default_hp_metric=False
            )
        logger_tb.save = lambda *args, **kwargs: None

        monitors = cfg_copy.train.hparams_search.monitors
        optuna_callback = PatchedPruningCallback(trial, monitor=monitors)
        trainer = pl.Trainer(
            max_epochs=cfg_copy.train.hparams_search.max_epochs,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            logger=logger_tb,
            enable_checkpointing=False,
            accumulate_grad_batches=grad_accum,
            callbacks=[optuna_callback],
            deterministic=cfg_copy.train.training_features.use_deterministic,
            enable_progress_bar=True,
            enable_model_summary=False,
            num_sanity_val_steps=0)

        logger.info(f"Starting Optuna trial {trial.number} with params {trial.params}")
        trainer.fit(model, datamodule=dm)
        result = tuple(trainer.callback_metrics[m].item() for m in monitors)
        formatted = ", ".join(f"{m}: {trainer.callback_metrics[m].item():.4f}" for m in monitors)
        logger.info(f"Trial {trial.number} results: \n {formatted}")
    except Exception as e:
        logger.error(f"Failed trial {trial.number} with error: {e} {traceback.format_exc()}")
        result = float("inf")

    return result


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

    src_train_path = db_path / (base_name + f'train_{keyword.lower()}.csv')
    src_valid_path = db_path / (base_name + f'valid_{keyword.lower()}.csv')
    src_test_path = db_path / (base_name + f'test_{keyword.lower()}.csv')
    ensure_dataset_files(src_train_path, src_valid_path, src_test_path)

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
    # Callbacks
    # ------------------------------------------------------------
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

    tb_save_dir = cfg_dict.logger.log_dir
    if not cfg_dict.general.save:
        tb_save_dir = tempfile.mkdtemp()
    else:
        main_logger.info(f"Tensorboard destination folder is {tb_save_dir}")
    with open_dict(cfg_dict):
        cfg_dict.tb_save_dir = tb_save_dir

    logger = TensorBoardLogger(
        save_dir=cfg_dict.tb_save_dir,
        name=None,
        default_hp_metric=False
        )
    logger.save = lambda *args, **kwargs: None
    safe_cfg_dict = recursive_clean(clean_cfg_dict)
    logger.hparams = {}
    logger.log_hyperparams(safe_cfg_dict, {"z_dummy": 0.0})

    # ------------------------------------------------------------
    # Optuna hyperparameters search
    # ------------------------------------------------------------
    if cfg_dict.train.hparams_search.use:
        main_logger.info("Starting hyperparameters search")
        study = optuna.create_study(directions=cfg_dict.train.hparams_search.directions)
        optuna_callbacks = [cb for cb in callbacks if not isinstance(cb, ModelCheckpoint)]

        main_logger.debug("Using monitors: " + ", ".join(cfg_dict.train.hparams_search.monitors))
        study.optimize(
            lambda trial: objective(main_logger, trial, cfg_dict, kwargs, optuna_callbacks),
            n_trials=cfg_dict.train.hparams_search.n_trials, catch=())
        best_trials = study.best_trials
        main_logger.info(f"Found {len(best_trials)} optimal trials")
        for i, t in enumerate(best_trials):
            main_logger.debug(f"Trial {t.number}: values={t.values}, params={t.params}")

        best_trial = min(best_trials, key=lambda t: 0.7*t.values[0] - 0.3*t.values[1])
        main_logger.info(f"Selected trial {best_trial.number} with values={best_trial.values}")
        # Update cfg_dict with best hyperparameters
        with open_dict(cfg_dict):
            for key, value in best_trial.params.items():
                parts = key.split(".")
                d = cfg_dict
                for p in parts[:-1]:
                    d = d[p]
                d[parts[-1]] = value

        load_logger_config(cfg_dict.logger)
    else:
        pass

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------
    main_logger.dev("Creating model")
    torch.set_float32_matmul_precision("medium")

    if cfg_dict.general.use_baseline:
        model = VAPModel(
            conf=base_conf,
            hyperparameters=cfg_dict.train.hyperparameters,
            lr_scheduler_config=cfg_dict.train.callbacks.lr_scheduler)
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
    # Training and validation data loaders
    # ------------------------------------------------------------

    main_logger.dev("Initializing training")

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
            profiler=None,
            enable_progress_bar=True,
            enable_model_summary=False,
            enable_checkpointing=False,
            deterministic=cfg_dict.general.use_seed)

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
        log_every_n_steps=1,
        profiler="simple", #simple, advancced, pytorch,
        precision="16-mixed",
        # progress_bar_refresh_rate=0,
        reload_dataloaders_every_n_epochs=False,
        enable_progress_bar=True,
        enable_model_summary=True,
        enable_checkpointing=cfg_dict.train.training_features.save_checkpoints,
        deterministic=cfg_dict.general.use_seed,
        benchmark=True, # Enables cuDNN auto-tuner for fastest convolution algorithms
        use_distributed_sampler=True,
        detect_anomaly=False,
        num_sanity_val_steps=0)

    trainer.fit(model, datamodule=dm, ckpt_path=checkpoints_path)
    trainer.validate(model, datamodule=dm)

    # ------------------------------------------------------------
    # Testing model
    # ------------------------------------------------------------
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()
