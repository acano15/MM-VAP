# -*- coding: utf-8 -*-
from typing import Optional, Tuple
from torch import Tensor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src.libs.logger.log import getLogger


class CEarlyStoppingCallback(EarlyStopping):
    """
    Lightning callback for customizing Early stopping.

    Args:
        modality (str): The modality to monitor, either 'validation' or 'training'.
        monitor (str): Metric to monitor for early stopping.
        min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
        patience (int): Number of epochs with no improvement after which training will be stopped.
        verbose (bool): Whether to log messages about early stopping.
        mode (str): One of {'min', 'max'}, determines whether the monitored metric should be minimized or maximized.
        strict (bool): Whether to strictly check for the existence of the monitored metric.
        check_finite (bool): Whether to check if the monitored metric is finite.
        stopping_threshold (Optional[float]): Threshold for stopping based on divergence.
        divergence_threshold (Optional[float]): Threshold for divergence detection.
        check_on_train_epoch_end (Optional[bool]): Whether to check early stopping at the end of training epochs.
    """

    def __init__(self, modality: str = "validation", monitor: str = "val_loss",
                 min_delta: float = 0.0, patience: int = 3, verbose: bool = False,
                 mode: str = "min", strict: bool = True, check_finite: bool = False,
                 stopping_threshold: Optional[float] = None,
                 divergence_threshold: Optional[float] = None,
                 check_on_train_epoch_end: Optional[bool] = None, log_rank_zero_only: bool = False):
        self._logger = getLogger(self.__class__.__name__)

        modality = (modality or "validation").lower().strip()
        if modality not in {"validation", "training"}:
            raise ValueError(f"modality must be 'validation' or 'training', got: {modality}")
        self._modality = modality

        if self._modality == "training" and check_on_train_epoch_end is None:
            check_on_train_epoch_end = True

        super().__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=verbose,
            mode=mode,
            strict=strict,
            check_finite=check_finite,
            stopping_threshold=stopping_threshold,
            divergence_threshold=divergence_threshold,
            check_on_train_epoch_end=check_on_train_epoch_end,
            log_rank_zero_only=log_rank_zero_only)

        self._logger.info("EarlyStopping initialized successfully")
        self._logger.debug(
            f"Parameters: modality={self._modality} monitor='{self.monitor}' "
            f"mode={self.mode} patience={self.patience} min_delta={self.min_delta} strict={self.strict} "
            f"check_finite={self.check_finite} stop_thr={self.stopping_threshold} div_thr={self.divergence_threshold} "
            f"check_on_train_epoch_end={self._check_on_train_epoch_end}"
            )

    def on_validation_end(self, trainer, pl_module) -> None:
        if self._modality == "training":
            return

        if self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        self._run_early_stopping_check(trainer)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        run_here = (self._modality == "training") or self._check_on_train_epoch_end
        if not run_here or self._should_skip_check(trainer):
            return
        self._run_early_stopping_check(trainer)

    @staticmethod
    def _rank_prefix(trainer) -> str:
        if getattr(trainer, "world_size", 1) > 1:
            rank = getattr(trainer, "global_rank", None)
            return f"[rank={rank}] " if rank is not None else ""
        return ""

    @staticmethod
    def _safe_item(x: Tensor) -> float:
        try:
            return float(x.detach().cpu().item())
        except Exception:
            return float("nan")

    def _run_early_stopping_check(self, trainer) -> None:
        logs = trainer.callback_metrics
        if trainer.fast_dev_run:
            self._logger.debug("[CEarlyStopping] fast_dev_run=True → skip early stopping check.")
            return
        if not self._validate_condition_metric(logs):
            # base class already warned/raised if strict
            self._logger.debug(
                f"[CEarlyStopping] metric '{self.monitor}' not in logs; skip this check."
            )
            return

        current = logs[self.monitor].squeeze()
        cur_val = self._safe_item(current)
        self._logger.debug(
            f"[CEarlyStopping] check @ epoch={trainer.current_epoch} step={trainer.global_step} "
            f"monitor={self.monitor} current={cur_val:.6f} best={self._safe_item(self.best_score):.6f} "
            f"wait_count={self.wait_count}"
        )

        should_stop, reason = self._evaluate_stopping_criteria(current)

        should_stop = trainer.strategy.reduce_boolean_decision(should_stop, all=False)
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop:
            self.stopped_epoch = trainer.current_epoch

        # Use our logger instead of Lightning's global logger
        if reason and self.verbose:
            prefix = self._rank_prefix(trainer)
            self._logger.info(prefix + reason)

        if should_stop:
            self._logger.info(f"STOP triggered at epoch={self.stopped_epoch} "
                              f"(monitor='{self.monitor}', mode={self.mode})")

    def _evaluate_stopping_criteria(self, current: Tensor) -> Tuple[bool, Optional[str]]:
        should_stop, reason = super()._evaluate_stopping_criteria(current)
        if reason:
            self._logger.debug(f"Stopping decision: {reason}")
        else:
            self._logger.debug(f"No stop. wait_count={self.wait_count}, "
                               f"best={self._safe_item(self.best_score):.6f}")
        return should_stop, reason

    @staticmethod
    def _log_info(trainer, message: str, log_rank_zero_only: bool) -> None:
        logger = getLogger("CEarlyStopping")
        rank = trainer.global_rank if trainer.world_size > 1 else None
        if rank is None or not log_rank_zero_only or rank == 0:
            prefix = f"[rank={rank}] " if rank is not None else ""
            logger.info(prefix + message)
