import math
import torch
from typing import Dict, Tuple, List, Optional, Union
from torchmetrics.classification import Accuracy, F1Score
from torchmetrics.metric import Metric
from rich.table import Table
from rich.panel import Panel
from rich import box

from .events import TurnTakingEvents
from .objective import ObjectiveVAP
from src.libs.logger.log import getLogger

TASK_TYPE = 'multiclass'
NUM_CLASSES = 2
AVERAGE_TYPE = 'weighted'


class CTurnTakingMetrics:
    def __init__(self, conf):
        super().__init__()

        self._conf = conf
        self.event_extractor = None
        self.event_extractor = TurnTakingEvents(conf)
        self._objective = ObjectiveVAP(bin_times=conf.bin_times, frame_hz=conf.frame_hz)

        self._deferred_console_tables = {}

        self._train_metrics = {}
        self._val_metrics = {}
        self._test_metrics = {}

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self._logger = getLogger(self.__class__.__name__)

    def init_metrics(self, split: str, register_fn=None):
        """
        Initializes metrics for a given split and optionally registers them.

        Args:
            split (str): One of 'train', 'val', or 'test'.
            register_fn (callable, optional): A function like model.register_module(name, module).
        """
        metrics = getattr(self, f"_{split}_metrics")
        if not metrics:
            metrics = self._create_metrics()
            if split == "train":
                self._train_metrics = metrics
            elif split == "val":
                self._val_metrics = metrics
            elif split == "test":
                self._test_metrics = metrics
            else:
                raise ValueError(f"Unknown split: {split}")

            if register_fn is not None:
                for name, events in metrics.items():
                    for event, metric in events.items():
                        strname = f"{split}_{name}_{event}"
                        self._logger.dev(f"Registering {strname} event metric")
                        register_fn(strname, metric)

    def _create_metrics(self) -> Dict[str, Dict[str, Metric]]:
        """
        Creates a dictionary of accuracy and F1 metrics for various turn-taking event types.

        Returns:
            Dict[str, Dict[str, Metric]]: Nested dictionary of metrics categorized by type and event.
        """
        metrics = {"acc": {}, "f1": {}}

        metrics["acc"]["hs"] = Accuracy(
            task=TASK_TYPE, num_classes=NUM_CLASSES, average=AVERAGE_TYPE).to(self.device)
        metrics["acc"]["hs_"] = Accuracy(
            task=TASK_TYPE, num_classes=NUM_CLASSES, average=None).to(self.device)
        metrics["acc"]["ls"] = Accuracy(
            task=TASK_TYPE, num_classes=NUM_CLASSES, average=AVERAGE_TYPE).to(self.device)
        metrics["acc"]["ls_"] = Accuracy(
            task=TASK_TYPE, num_classes=NUM_CLASSES, average=None).to(self.device)
        metrics["acc"]["sp"] = Accuracy(
            task=TASK_TYPE, num_classes=NUM_CLASSES, average=AVERAGE_TYPE).to(self.device)
        metrics["acc"]["bp"] = Accuracy(
            task=TASK_TYPE, num_classes=NUM_CLASSES, average=AVERAGE_TYPE).to(self.device)

        metrics["f1"]["hs"] = F1Score(
            task=TASK_TYPE, num_classes=NUM_CLASSES, average=AVERAGE_TYPE).to(self.device)
        metrics["f1"]["hs_"] = F1Score(
            task=TASK_TYPE, num_classes=NUM_CLASSES, average=None).to(self.device)
        metrics["f1"]["ls"] = F1Score(
            task=TASK_TYPE, num_classes=NUM_CLASSES, average=AVERAGE_TYPE).to(self.device)
        metrics["f1"]["ls_"] = F1Score(
            task=TASK_TYPE, num_classes=NUM_CLASSES, average=None).to(self.device)
        metrics["f1"]["sp"] = F1Score(
            task=TASK_TYPE, num_classes=NUM_CLASSES, average=AVERAGE_TYPE).to(self.device)
        metrics["f1"]["bp"] = F1Score(
            task=TASK_TYPE, num_classes=NUM_CLASSES, average=AVERAGE_TYPE).to(self.device)

        if self._conf.lid_classify:
            metrics["f1"]["lid"] = F1Score(
                task=TASK_TYPE, num_classes=NUM_CLASSES, average=AVERAGE_TYPE).to(self.device)

        return metrics

    def metrics_step(self, batch, out, split: str = ""):
        """
        Converts model output into turn-taking predictions and updates metrics.

        Args:
            batch (dict): A batch from the dataloader, must include 'vad'.
            out (dict): Model output, must include 'logits'.
            split (str): One of 'train', 'val', or 'test' to select metrics.
        """
        events = self.event_extractor(batch["vad"])

        probs = self._objective.get_probs(out["logits"])
        preds, targets = self._objective.extract_prediction_and_targets(
            p_now=probs["p_now"], p_fut=probs["p_future"], events=events)
        self._update_metrics(preds, targets, split=split)
        return preds, targets

    def _update_metrics(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor],
                        split: Optional[str] = None) -> None:
        """
        Updates the accuracy and F1 metrics with the given predictions and targets.

        Args:
            preds (dict): Predicted outputs for different event types.
            targets (dict): Ground truth labels for corresponding events.
            split (str, optional): Dataset split ('train', 'val', or 'test') used to select metric set.
        """

        if split == "train":
            m = self._train_metrics
        elif split == "val":
            m = self._val_metrics
        else:
            m = self._test_metrics

        if preds["hs"] is not None:
            if len(targets["hs"]) >= 1:
                m["f1"]["hs"].update(preds=preds["hs"].round().int(), target=targets["hs"].int())
                m["acc"]["hs"].update(preds=preds["hs"].round().int(), target=targets["hs"].int())
                m["f1"]["hs_"].update(preds=preds["hs"].round().int(), target=targets["hs"].int())
                m["acc"]["hs_"].update(preds=preds["hs"].round().int(), target=targets["hs"].int())

        if preds["ls"] is not None:
            if len(targets["ls"]) >= 1:
                m["f1"]["ls"].update(preds=preds["ls"].round().int(), target=targets["ls"].int())
                m["acc"]["ls"].update(preds=preds["ls"].round().int(), target=targets["ls"].int())
                m["f1"]["ls_"].update(preds=preds["ls"].round().int(), target=targets["ls"].int())
                m["acc"]["ls_"].update(preds=preds["ls"].round().int(), target=targets["ls"].int())

        if preds["pred_shift"] is not None:
            if len(targets["pred_shift"]) >= 1:
                m["f1"]["sp"].update(
                    preds=preds["pred_shift"].round().int(), target=targets["pred_shift"].int())
                m["acc"]["sp"].update(
                    preds=preds["pred_shift"].round().int(), target=targets["pred_shift"].int())

        if preds["pred_backchannel"] is not None:
            if len(targets["pred_backchannel"]) >= 1:
                m["f1"]["bp"].update(
                    preds=preds["pred_backchannel"].round().int(), target=targets["pred_backchannel"].int())
                m["acc"]["bp"].update(
                    preds=preds["pred_backchannel"].round().int(), target=targets["pred_backchannel"].int())

        if self._conf.lid_classify and preds["lid"] is not None:
            if len(targets["lid"]) >= 1:
                m["f1"]["lid"].update(preds=preds["lid"], target=targets["lid"])

    def metrics_epoch(
        self, split: str = "val", batch_size: int = 1,
        epoch: Optional[int] = None) -> dict:
        """
        Computes and logs metrics at the end of an epoch for a given split.

        Args:
            split (str): One of 'train', 'val', or 'test'.
            batch_size (int): Not used in current implementation.
            epoch (int, optional): Current epoch number, for display/logging.
        Returns:
            Dict containing metrics.
        """

        if split == "train":
            metrics = self._train_metrics
        elif split == "val":
            metrics = self._val_metrics
        else:
            metrics = self._test_metrics

        candidate_dict = {
            "acc_{}_hs".format(split): metrics['acc']['hs'],
            "acc_{}_hs_".format(split): metrics['acc']['hs_'],
            "acc_{}_ls".format(split): metrics['acc']['ls'],
            "acc_{}_ls_".format(split): metrics['acc']['ls_'],
            "acc_{}_sp".format(split): metrics['acc']['sp'],
            "acc_{}_bp".format(split): metrics['acc']['bp'],
            "f1_{}_hs".format(split): metrics['f1']['hs'],
            "f1_{}_ls".format(split): metrics['f1']['ls'],
            "f1_{}_sp".format(split): metrics['f1']['sp'],
            "f1_{}_bp".format(split): metrics['f1']['bp'],
            }
        if self._conf.lid_classify:
            candidate_dict["f1_{}_lid".format(split)] = metrics['f1']['lid']

        display_metrics = {}
        for key, metric in candidate_dict.items():
            if isinstance(metric, Metric):
                try:
                    if getattr(metric, "_update_called", False):
                        tmp_result = metric.compute()
                        if tmp_result is not None:
                            if isinstance(tmp_result, torch.Tensor) and tmp_result.ndim == 1:
                                for i, val in enumerate(tmp_result):
                                    # Map classes explicitly for HS and LS
                                    if "hs_" in key:
                                        class_name = "hold" if i == 0 else "shift"
                                    elif "ls_" in key:
                                        class_name = "short" if i == 0 else "long"
                                    else:
                                        class_name = f"class{i}"

                                    class_key = f"{key[:-4]}_{class_name}"
                                    display_metrics[class_key] = float(val.item())
                                    self._logger.info(
                                        f"Epoch {epoch}: {split} {class_key} = {val.item():.4f}")
                            else:
                                display_metrics[key] = float(tmp_result)
                                self._logger.info(
                                    f"Epoch {epoch}: {split} {key} = {float(tmp_result):.4f}")
                        else:
                            self._logger.debug(f"No result for {key}")
                    else:
                        self._logger.debug(f"Update not called for {key}")
                except (ValueError, RuntimeError) as e:
                    self._logger.error(f"Error in metric {key}: {e}")

        if display_metrics:
            panel = self._build_metrics_panel(split, display_metrics, epoch)
            self._deferred_console_tables[split] = panel

        return display_metrics

    def reset(self) -> None:
        """
        Resets all stored metrics across train, validation, and test splits.
        Useful at the beginning of each epoch or evaluation cycle.
        """

        self._logger.debug("Resetting metrics")

        self._deferred_console_tables = {}
        for metric in ["acc", "f1"]:
            for key in self._train_metrics[metric].keys():
                if self._train_metrics and key in self._train_metrics[metric]:
                    self._train_metrics[metric][key].reset()

                if self._val_metrics and key in self._val_metrics[metric].keys():
                    self._val_metrics[metric][key].reset()

                if self._test_metrics and key in self._test_metrics[metric].keys():
                    self._test_metrics[metric][key].reset()

    def get_deferred_console_tables(self, split: Optional[str] = None) -> List[Panel]:
        """
        Returns a list of deferred console tables for display.

        Args:
            split (str, optional): If provided, returns tables only for this split.
        """
        tables = []
        if split == "test":
            tables = self._deferred_console_tables.get("test", [])
        else:
            for split in ["train", "val"]:
                if split in self._deferred_console_tables:
                    tables.append(self._deferred_console_tables[split])
                    del self._deferred_console_tables[split]

        return tables

    @staticmethod
    def _build_metrics_panel(split: str, display_metrics: dict, epoch: int) -> Panel:
        """
        Builds a rich console panel displaying the current epoch's metrics.

        Args:
            split (str): Data split ('train', 'val', or 'test').
            display_metrics (dict): Dictionary of metric names to computed values.
            epoch (int): Current epoch number.

        Returns:
            Panel: A rich formatted panel showing the metrics.
        """
        table = Table(title=f"{split} metrics (epoch {epoch})", box=box.SIMPLE_HEAD)
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        for k, v in sorted(display_metrics.items()):
            table.add_row(k, f"{v:.4f}")

        return Panel(table, padding=(0, 1), border_style="cyan")
