# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import Image
import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from src.libs.logger.log import getLogger


class CMetricPlotCallback(Callback):
    def __init__(self, save_dir="plots", log_to_tensorboard=True):
        self.save_dir = save_dir
        self.log_to_tensorboard = log_to_tensorboard
        self.history = defaultdict(list)
        self._logger = getLogger(self.__class__.__name__)

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        for k, v in trainer.logged_metrics.items():
            if "train" in k and k.startswith("step"):
                self.history[k].append(v.item())

    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        for k, v in trainer.logged_metrics.items():
            if "val" in k and k.startswith("step"):
                self.history[k].append(v.item())

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        self._collect_epoch_metrics(trainer, prefix="train")

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        self._collect_epoch_metrics(trainer, prefix="val")
        self._plot_metrics(trainer)

    def _collect_epoch_metrics(self, trainer, prefix="train"):
        for k, v in trainer.callback_metrics.items():
            if k.startswith(prefix) and k.endswith("_epoch"):
                self.history[k].append(v.item())

    def _plot_metrics(self, trainer):
        os.makedirs(self.save_dir, exist_ok=True)
        step_metrics = defaultdict(list)
        epoch_metrics = defaultdict(list)

        # Split step/epoch metrics
        for metric, values in self.history.items():
            if metric.endswith("_step"):
                step_metrics[metric].extend(values)
            elif metric.endswith("_epoch"):
                epoch_metrics[metric].extend(values)

        # === Epoch-level plots ===
        for base_name in self._group_metric_names(epoch_metrics.keys()):
            plt.figure()
            for full_name in base_name["full_names"]:
                values = self.history[full_name]
                label = full_name.replace("_epoch", "")
                plt.plot(values, label=label)
            plt.title(f"Epoch Metric: {base_name['name']}")
            plt.xlabel("Epoch")
            plt.ylabel("Value")
            plt.legend()
            self._save_plot(trainer, metric_name=base_name["name"])

        # === Step-level plots (optional) ===
        for base_name in self._group_metric_names(step_metrics.keys()):
            plt.figure()
            for full_name in base_name["full_names"]:
                values = self.history[full_name]
                label = full_name.replace("_step", "")
                plt.plot(values, label=label, alpha=0.6)
            plt.title(f"Step Metric: {base_name['name']}")
            plt.xlabel("Steps")
            plt.ylabel("Value")
            plt.legend()
            self._save_plot(trainer, metric_name=base_name["name"])

    def _group_metric_names(self, keys):
        # Groups 'train_loss_step', 'val_loss_step' into {'name': 'loss_step', 'full_names': [...]}
        grouped = defaultdict(list)
        for k in keys:
            base = k.replace("/", "_").replace("train_", "").replace("val_", "")
            grouped[base].append(k)
        return [{"name": k, "full_names": v} for k, v in grouped.items()]

    def _save_plot(self, trainer, metric_name):
        epoch = trainer.current_epoch
        time_str = datetime.datetime.now().strftime("%H_%M_%S")
        filename = f"epoch{epoch}_{metric_name}_{time_str}.png"
        path = os.path.join(self.save_dir, filename)
        self._logger.info(f"Saving behavior image {path}")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

        # Optional: log to TensorBoard
        if self.log_to_tensorboard and hasattr(trainer.logger, "experiment"):
            trainer.logger.experiment.add_image(
                tag=filename,
                img_tensor=self._load_image_for_tensorboard(path),
                global_step=trainer.global_step,
                dataformats="HWC"
            )

    def _load_image_for_tensorboard(self, path):
        img = Image.open(path).convert("RGB")
        return np.array(img)
