# -*- coding: utf-8 -*-
import sys
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn
)
from rich.console import Console
import logging

from src.libs.logger.log import getLogger


class _EpochLogBufferHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET, formatter: logging.Formatter = None, console: Console = None):
        super().__init__(level)
        self._fmt = formatter
        self._buf = []
        self._console = console or Console(file=sys.__stdout__, markup=False, highlight=False)

    def emit(self, record: logging.LogRecord) -> None:
        msg = self._fmt.format(record) if self._fmt else record.getMessage()
        if record.levelno >= logging.WARNING or record.levelno < logging.DEBUG:
            self._console.print(msg, markup=False, highlight=False, soft_wrap=True)
        else:
            self._buf.append(msg)

    def pop_all(self) -> str:
        out = "\n".join(self._buf)
        self._buf.clear()
        return out


class CCustomProgressBar(RichProgressBar):
    def __init__(self):
        super().__init__(theme=RichProgressBarTheme())
        self._console = Console(file=sys.__stdout__, markup=False, highlight=False)

        self._silenced_console_handlers = []
        self._log_buffers = []
        self._stdout_orig = None
        self._stdout_proxy = None
        self._progress = Progress(
            TextColumn("[bold blue]{task.fields[phase]} | Epoch: {task.fields[epoch_display]} | Step: {task.completed}/{task.total}", justify="right"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[metrics]}", justify="left"),
            console=self._console,
            transient=False,
        )
        self._logger = getLogger(self.__class__.__name__)

        self._tasks = {}
        self._trainer = None
        self._max_epochs = None
        self._log_stream_proxy = None
        self._patched_handlers = []

    def setup(self, trainer, pl_module, stage: str):
        self._trainer = trainer
        self._max_epochs = trainer.max_epochs
        self._install_epoch_log_buffers()

        self._progress.start()

    def _add_rich_task(self, name, total):
        if name in self._tasks:
            self._progress.remove_task(self._tasks[name])
        self._tasks[name] = self._progress.add_task(
            "",
            total=total,
            completed=0,
            phase=name,
            epoch_display=f"{self._trainer.current_epoch + 1}/{self._max_epochs}",
            metrics=""
        )

    def _get_filtered_metrics(self, trainer, phase, final=False):
        phase_key = phase.lower()
        return {
            k: v for k, v in trainer.progress_bar_metrics.items()
            if (
                isinstance(v, (float, int)) and
                k.startswith(phase_key) and
                (
                    final and k.endswith("_epoch") or
                    not final and k.endswith("_step")
                )
            )
            }

    def _update(self, phase, batch_idx, trainer, final=False, visible=None):
        task_id = self._tasks.get(phase)
        if task_id is None:
            return

        metrics = self._get_filtered_metrics(trainer, phase, final=final)
        metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())

        self._progress.update(
            task_id,
            completed=batch_idx + 1,
            epoch_display=f"{trainer.current_epoch + 1}/{self._max_epochs}",
            metrics=metrics_str,
            visible=visible
            )
        self._progress.refresh()

    def on_train_epoch_start(self, trainer, pl_module):
        self._flush_epoch_logs_to_console()
        self._console.print("")  # newline between epochs
        if "Val" in self._tasks:
            self._progress.remove_task(self._tasks["Val"])
            del self._tasks["Val"]
        self._add_rich_task("Train", trainer.num_training_batches)

    def on_validation_epoch_start(self, trainer, pl_module):
        self._add_rich_task("Val", trainer.num_val_batches[0])

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._update("Train", batch_idx, trainer)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._update("Val", batch_idx, trainer)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._update("Val", trainer.num_val_batches[0] - 1, trainer, final=True)
        self._progress.refresh()

        for k in list(trainer.progress_bar_metrics):
            if "val_loss" in k or k.endswith("_step") or k.endswith("_epoch"):
                del trainer.progress_bar_metrics[k]

    def on_train_epoch_end(self, trainer, pl_module):
        self._update("Train", trainer.num_training_batches - 1, trainer, final=True)
        self._progress.refresh()
        # Clear step metrics
        for k in list(trainer.progress_bar_metrics):
            if "train_loss" in k or k.endswith("_step") or k.endswith("_epoch"):
                del trainer.progress_bar_metrics[k]

    def on_test_epoch_start(self, trainer, pl_module):
        for name, task_id in list(self._tasks.items()):
            self._progress.remove_task(task_id)

        self._tasks.clear()
        self._tasks["Test"] = self._progress.add_task(
            "",
            total=trainer.num_test_batches[0],
            completed=0,
            phase="Test",
            epoch_display=f"{self._trainer.current_epoch + 1}/{self._max_epochs}",
            metrics=""
            )

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._update("Test", batch_idx, trainer)

    def on_test_epoch_end(self, trainer, pl_module):
        self._flush_epoch_logs_to_console()

    def teardown(self, trainer, pl_module, stage: str):
        if len(self._log_buffers) > 0:
            self._flush_epoch_logs_to_console()
        self._progress.stop()
        self._remove_epoch_log_buffers()
        self._restore_console_handlers()

    def _install_epoch_log_buffers(self):
        targets = [logging.getLogger()] + [
            logging.getLogger(n)
            for n in logging.root.manager.loggerDict.keys()
            if isinstance(logging.getLogger(n), logging.Logger)
            ]
        for lg in targets:
            # find console handlers only (keep file handlers untouched)
            chs = [
                h for h in getattr(lg, "handlers", [])
                if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
                ]
            if not chs:
                continue
            # use the first console handler's level & formatter (keeps your colorlog colors and thresholds)
            base = chs[0]
            buf = _EpochLogBufferHandler(level=base.level, formatter=base.formatter, console=self._console)
            lg.addHandler(buf)
            self._log_buffers.append((lg, buf))
            # silence existing console handlers during the epoch (so they don't break the bar)
            for h in chs:
                self._silenced_console_handlers.append((lg, h, h.level))
                h.setLevel(10_000)

    def _restore_console_handlers(self):
        for lg, h, lvl in self._silenced_console_handlers:
            try:
                h.setLevel(lvl)
            except Exception:
                pass
        self._silenced_console_handlers.clear()

    def _flush_epoch_logs_to_console(self):
        if not self._log_buffers:
            pass
        else:
            lines = []
            for lg, buf in self._log_buffers:
                text = buf.pop_all()
                if text:
                    lines.append(text)
            if lines:
                self._console.print("")  # newline separator
                self._console.print("\n".join(lines), markup=False, highlight=False, soft_wrap=True)

        self._clear_epoch_log_buffers()

    def _remove_epoch_log_buffers(self):
        for lg, buf in self._log_buffers:
            try:
                lg.removeHandler(buf)
            except Exception:
                pass
        self._log_buffers.clear()

    def _clear_epoch_log_buffers(self):
        for _, buf in getattr(self, "_log_buffers", []):
            buf.pop_all()


