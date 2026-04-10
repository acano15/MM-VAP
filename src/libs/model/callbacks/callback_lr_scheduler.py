# -*- coding: utf-8 -*-
from typing import Any, Optional, Dict
import torch
import torch.optim.lr_scheduler as lr_scheduler
from transformers import get_polynomial_decay_schedule_with_warmup
import pytorch_lightning as pl

from src.libs.logger.log import getLogger


class CLearningRateSchedulerCallback(pl.Callback):
    """
    Custom LR Scheduler wrapper for Lightning with logging.

    Args:
        optimizer: Optimizer used for training.
        num_training_steps: Total number of training steps.
        config: Scheduler configuration dictionary.
        step_interval: Scheduler stepping granularity ("step" or "epoch").
        monitor: Metric name monitored by ReduceLROnPlateau.
        warmup_steps_rate: Rate of warmup steps relative to total training steps.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_training_steps: int,
        config: Dict[str, Any],
        step_interval: str = "epoch",
        monitor: str = "eval_loss",
        warmup_steps_rate: float = 0.2,
    ) -> None:
        self._logger = getLogger(self.__class__.__name__)

        self._optimizer = optimizer
        self._num_training_steps = num_training_steps
        self._config = config
        self._step_interval = step_interval
        self._monitor = monitor
        self._warmup_steps = int(
            self._config.get("warmup_steps_rate", warmup_steps_rate) * self._num_training_steps
            )

        self._scheduler = self._build_scheduler()

        if "initial_lr" in self._optimizer.param_groups[0].keys():
            self._last_lr = float(self._optimizer.param_groups[0]["initial_lr"])
        else:
            self._last_lr = float(self._optimizer.param_groups[0]["lr"])
        self._logger.info(f"Learning Rate Scheduler {self._config.type} initialized. Initial LR: "
                          f"{self._last_lr} with {self._step_interval} strategy")

    def get_scheduler(self) -> lr_scheduler._LRScheduler:
        """
        Return the created learning-rate scheduler.
        """
        return self._scheduler

    def _build_scheduler(self):
        """
        Build and return a torch learning-rate scheduler from configuration.
        """
        sch_type = self._config.get("type", None)
        params = self._config

        if sch_type == "LinearLR":
            sched = get_polynomial_decay_schedule_with_warmup(
                self._optimizer,
                num_warmup_steps=self._warmup_steps,
                num_training_steps=self._num_training_steps,
                lr_end=params.get("lr_end", 0.0),
                power=params.get("power", 1.0),
            )
            self._step_interval = "step"
        elif sch_type == "ReduceLROnPlateau":
            sched = lr_scheduler.ReduceLROnPlateau(
                optimizer=self._optimizer,
                mode=params.get("mode", "min"),
                factor=params.get("factor", 0.1),
                patience=params.get("patience", 10),
                threshold=params.get("threshold", 1e-4),
                min_lr=params.get("min_lr", 0.0),
                verbose=False,
            )
            self._step_interval = "epoch"
        elif sch_type == "CosineAnnealingLR":
            sched = lr_scheduler.CosineAnnealingLR(
                optimizer=self._optimizer,
                T_max=params.get("T_max", 50),
                eta_min=params.get("eta_min", 0.0),
            )
            self._step_interval = "epoch"
        elif sch_type == "StepLR":
            sched = lr_scheduler.StepLR(
                optimizer=self._optimizer,
                step_size=params.get("step_size", 10),
                gamma=params.get("gamma", 0.1),
            )
            self._step_interval = "epoch"
        elif sch_type == "CyclicLR":
            sched = lr_scheduler.CyclicLR(
                optimizer=self._optimizer,
                base_lr=params.get("base_lr", 1e-4),
                max_lr=params.get("max_lr", 0.1),
                step_size_up=params.get("step_size_up", 2000),
                step_size_down=params.get("step_size_down", 2000),
                mode=params.get("modeCLR", "triangular2"),
                cycle_momentum=params.get("cycle_momentum", True),
                base_momentum=params.get("base_momentum", 0.8),
                max_momentum=params.get("max_momentum", 0.9),
            )
            self._step_interval = "step"
        elif sch_type == "ExponentialLR":
            sched = lr_scheduler.ExponentialLR(
                optimizer=self._optimizer,
                gamma=params.get("gamma", 0.95))
            self._step_interval = "epoch"
        elif sch_type == "MultiStepLR":
            sched = lr_scheduler.MultiStepLR(
                optimizer=self._optimizer,
                milestones=params.get("milestones", [30, 60, 90]),
                gamma=params.get("gamma", 0.1))
            self._step_interval = "epoch"
        elif sch_type == "LambdaLR":
            scale_fn = params.get("scale_fn", None)
            if scale_fn is None or scale_fn == "None":
                scale_fn = lambda epoch: 1.0
            sched = lr_scheduler.LambdaLR(
                optimizer=self._optimizer,
                lr_lambda=scale_fn)
            self._step_interval = "epoch"
        else:
            raise ValueError(f"Unknown scheduler type: {sch_type}")

        return sched

    def lightning_dict(self):
        if isinstance(self._scheduler, lr_scheduler.ReduceLROnPlateau):
            monitor = self._config.get("monitor", "val_loss")
            return {
                "scheduler": self._scheduler,
                "monitor": monitor,
                "interval": "epoch",
                "frequency": 1,
            }
        else:
            return {
                "scheduler": self._scheduler,
                "interval": self._step_interval,
                "frequency": 1,
            }

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._config.get("type") != "ReduceLROnPlateau" and self._step_interval == "step":
            self._check_and_log()

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if self._config.get("type") != "ReduceLROnPlateau" and self._step_interval == "epoch":
            self._check_and_log()

    def on_validation_epoch_end(self, trainer, pl_module):
        # ReduceLROnPlateau is stepped on monitored metric (typically after validation)
        if self._config.get("type") == "ReduceLROnPlateau":
            value = trainer.callback_metrics.get(self._monitor)
            if value is not None:
                self._check_and_log()

    def _current_lr(self) -> float:
        """
        Return current learning rate of the first parameter group.
        """
        return float(self._optimizer.param_groups[0]["lr"])

    def _check_and_log(self):
        """Check learning rate, log each epoch and detect changes."""
        current_lr = self._current_lr()

        if current_lr != self._last_lr:
            lr_log_msg = f"Learning rate changed from {self._last_lr:.10f} to {current_lr:.10f}"
            if self._step_interval == "step":
                self._logger.debug(lr_log_msg)
            elif self._step_interval == "epoch":
                self._logger.info(lr_log_msg)
            self._last_lr = current_lr
        else:
            self._logger.debug(f"No changes in the current LR: {current_lr:.10f}")
