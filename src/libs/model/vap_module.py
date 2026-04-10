import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim
from typing import Dict, Tuple, List, Optional, Union
from rich.console import Console
from rich.columns import Columns
from io import StringIO
from pytorch_lightning.callbacks.progress import ProgressBar
from omegaconf import OmegaConf, DictConfig
import logging

from .multimodal_model import CMultimodalVAP
from src.libs.events.turn_taking_metrics import CTurnTakingMetrics
from .callbacks import CCustomMetric
from .callbacks import CLearningRateSchedulerCallback
from src.libs.logger.log import getLogger


# Definitions
TRAIN = "train"
VAL = "val"
TEST = "test"
BIN_TIMES: list = [0.2, 0.4, 0.6, 0.8]


class CVAPModule(CMultimodalVAP, pl.LightningModule):
    def __init__(self, conf: dict | OmegaConf = None, hyperparameters: Optional[Dict] = None,
                 lr_scheduler_config: Optional[Dict] = None):
        self._logger = getLogger(self.__class__.__name__)

        super().__init__(conf)
        self.model_conf = self._conf
        self.hyperparameters = hyperparameters
        self.lr_scheduler_config = lr_scheduler_config

        self.lr = self.hyperparameters.learning_rate
        # Training params
        self.save_hyperparameters()

        # Metrics
        self._turn_metrics_callback = None
        if not self.hyperparameters.find_learning_rate:
            self._turn_metrics_callback = CTurnTakingMetrics(conf.events_configuration)

        self._use_val = False
        self._custom_metrics_vad = None
        self._custom_metrics_vap = None
        if not self.hyperparameters.find_learning_rate:
            self._custom_metrics_vad = {TRAIN: CCustomMetric(),
                                    VAL: CCustomMetric(),
                                    TEST: CCustomMetric()}
            self._custom_metrics_vap = {TRAIN: CCustomMetric(),
                                    VAL: CCustomMetric(),
                                    TEST: CCustomMetric()}
        self._logger = getLogger(self.__class__.__name__)

    def load_pretrained_parameters(self, vap_model, cpc_model, face_encoder_path=None):
        self._logger.info("Loading pretrained VAP models")
        sd = torch.load(vap_model, map_location=torch.device(self.device))

        # Remove training-related keys
        sd.pop('global_step', None)
        sd.pop('epoch', None)

        model_params = dict(self.named_parameters())
        loaded, skipped, adapted = 0, 0, 0

        for name, param in sd.items():
            if name.startswith(
                ("zero_shot.", "va_classifier", "vap_head", "lid_classifier",
                 "lid_classifier_middle")):
                self._logger.debug(f"Skipping layer: {name}")
                skipped += 1
                continue

            if name in model_params:
                tgt = model_params[name]
                if param.shape == tgt.shape:
                    self._logger.debug(f"Loading layer: {name}")
                    with torch.no_grad():
                        tgt.copy_(param.to(device=tgt.device, dtype=tgt.dtype))
                    loaded += 1
                elif tgt.ndim == 3 and param.ndim == 3 and tgt.shape[:2] == param.shape[:2]:
                    # Adapt conv kernel size
                    k_model, k_ckpt = tgt.shape[2], param.shape[2]
                    if k_model < k_ckpt:
                        param = param[..., :k_model]
                    else:
                        pad = k_model - k_ckpt
                        param = F.pad(param, (0, pad))
                    with torch.no_grad():
                        tgt.copy_(param.to(device=tgt.device, dtype=tgt.dtype))
                    adapted += 1
                    self._logger.debug(f"Adapted kernel size for {name}")
                else:
                    self._logger.warning(
                        f"Shape mismatch for {name}: model {tgt.shape}, ckpt {param.shape}")
                    skipped += 1
            else:
                if self._conf.use_backbone and (
                    name.startswith("ar") or name.startswith("encoder.")):
                    continue

                self._logger.warning(f"Unknown parameter in checkpoint: {name}")
                skipped += 1

        self._logger.info(
            f"Loaded {loaded}, adapted {adapted}, skipped {skipped} parameters.")

    def configure_optimizers(self) -> Dict:
        assert self.hyperparameters is not None, "configure_optimizers: No hyperparameters conf!"

        result = {}
        self._logger.debug("Preparing hyperparameters")
        if self.hyperparameters.use_adopt:
            from adopt import ADOPT
            opt = ADOPT(self.parameters(), lr=self.hyperparameters.learning_rate, decouple=True)
        else:
            opt = self._build_optimizer()

        result["optimizer"] = opt
        self._lr_scheduler_wrapper = None
        if self.lr_scheduler_config.use:
            num_steps = int(self.trainer.estimated_stepping_batches)
            self._lr_scheduler_wrapper = CLearningRateSchedulerCallback(
                optimizer=opt,
                num_training_steps=num_steps,
                config=self.lr_scheduler_config,
                step_interval=getattr(self.lr_scheduler_config, "step_interval", "step"),
                monitor=getattr(self.lr_scheduler_config, "monitor", "eval_loss"),
                warmup_steps_rate=getattr(self.lr_scheduler_config, "warmup_steps_rate", 0.2),
                )
            result["lr_scheduler"] = self._lr_scheduler_wrapper.lightning_dict()

        return result

    def shared_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Arguments:
            batch:      dict, containing 'waveform', va, va_history

        Returns:
            out:        dict, ['logits', 'vad']
        """
        src_input = {}
        labels = self.objective.get_labels(batch["vad"].to(self.device))
        src_input["labels"] = labels
        src_input["waveform"] = batch["waveform"].to(self.device)

        if self._conf.use_face_encoder or self._conf.use_backbone:
            src_input["face_im1"] = batch["face_im1"].to(self.device)
            src_input["face_im2"] = batch["face_im2"].to(self.device)

        if self._conf.multimodal:
            src_input["gaze1"] = batch["gaze1"].to(self.device)
            src_input["head1"] = batch["head1"].to(self.device)
            src_input["face1"] = batch["face1"].to(self.device)
            src_input["body1"] = batch["body1"].to(self.device)
            src_input["gaze2"] = batch["gaze2"].to(self.device)
            src_input["head2"] = batch["head2"].to(self.device)
            src_input["face2"] = batch["face2"].to(self.device)
            src_input["body2"] = batch["body2"].to(self.device)

        for k, v in src_input.items():
            if isinstance(v, torch.Tensor):
                self._logger.dev(f"src_input[{k}] shape: {tuple(v.shape)}")

        # Forward
        out = self(src=src_input)
        out["vad_labels"] = batch["vad"]
        out["labels"] = labels
        return out

    def on_fit_start(self):
        if self._lr_scheduler_wrapper is not None and self._lr_scheduler_wrapper not in self.trainer.callbacks:
            self.trainer.callbacks.append(self._lr_scheduler_wrapper)

    def on_train_start(self):
        self._use_val = False
        if self._turn_metrics_callback is not None:
            self._turn_metrics_callback.init_metrics(TRAIN, self.register_module)

    def training_step(self, batch, batch_idx, **kwargs):
        out = self.shared_step(batch)

        out["loss_vap"] = self.objective.loss_vap(out["logits"], out["labels"],
                                                           reduction="mean")
        out["loss_vad"] = self.objective.loss_vad(out["vad_logits"], out["vad_labels"])
        loss_vap = out["loss_vap"].detach()
        loss_vad = out["loss_vad"].detach()

        semantic_regularizer = self.objective.loss_semantic_regularization(out["logits"],
                                                                           out["vad_labels"])

        loss = out["loss_vap"] + out["loss_vad"] + 0.2 * semantic_regularizer
        self._log_metric("loss_vap", loss_vap, on_step=True, on_epoch=True, batch_idx=batch_idx)
        self._log_metric("loss_vad", loss_vad, on_step=True, on_epoch=True, batch_idx=batch_idx)
        self._log_metric("loss", loss, on_step=True, on_epoch=True, batch_idx=batch_idx)

        # Metrics
        if self._custom_metrics_vad is not None:
            probs_vad = torch.sigmoid(out["vad_logits"])   # [B,T,2]
            preds_vad = (probs_vad > 0.5).int()
            n = out["vad_logits"].shape[-2]
            vad_labels = batch["vad"][:, :n, :].float()  # [B,T,2]
            self._custom_metrics_vad[TRAIN].update(preds_vad, vad_labels, loss_vad, probs_vad)

        if self._custom_metrics_vap is not None:
            probs_vap = torch.softmax(out["logits"], dim=-1)
            preds_vap = probs_vap.argmax(dim=-1)
            self._custom_metrics_vap[TRAIN].update(preds_vap, out["labels"], loss_vap, probs_vap)

        if self._turn_metrics_callback is not None:
            turntaking_preds, turntaking_targets = self._turn_metrics_callback.metrics_step(
                batch, out, split=TRAIN)

        return {"loss": loss}

    def on_validation_start(self):
        self._use_val = True
        if self._turn_metrics_callback is not None:
            self._turn_metrics_callback.init_metrics(VAL, self.register_module)

    def validation_step(self, batch, batch_idx, **kwargs):
        """validation step"""
        out = self.shared_step(batch)

        out["loss_vap"] = self.objective.loss_vap(out["logits"], out["labels"], reduction="mean")
        out["loss_vad"] = self.objective.loss_vad(out["vad_logits"], out["vad_labels"])
        loss_vap = out["loss_vap"].detach()
        loss_vad = out["loss_vad"].detach()
        val_loss = out["loss_vap"] + out["loss_vad"]
        self._log_metric("loss_vap", loss_vap, on_step=True, on_epoch=True, batch_idx=batch_idx)
        self._log_metric("loss_vad", loss_vad, on_step=True, on_epoch=True, batch_idx=batch_idx)
        self._log_metric("loss", val_loss, on_step=True, on_epoch=True, batch_idx=batch_idx)

        # Metrics
        if self._custom_metrics_vad is not None:
            probs_vad = torch.sigmoid(out["vad_logits"])   # [B,T,2]
            preds_vad = (probs_vad > 0.5).int()
            n = out["vad_logits"].shape[-2]
            vad_labels = batch["vad"][:, :n, :].float()  # [B,T,2]
            self._custom_metrics_vad[VAL].update(preds_vad, vad_labels, loss_vad, probs_vad)

        if self._custom_metrics_vap is not None:
            probs_vap = torch.softmax(out["logits"], dim=-1)
            preds_vap = probs_vap.argmax(dim=-1)
            self._custom_metrics_vap[VAL].update(preds_vap, out["labels"], loss_vap, probs_vap)

        if self._turn_metrics_callback is not None:
            val_turntaking_preds, val_turntaking_targets = self._turn_metrics_callback.metrics_step(
                batch, out, split=VAL)

    def on_validation_epoch_end(self, *_):
        if self._custom_metrics_vad is not None:
            custom_metric_vad = self._custom_metrics_vad[VAL]
            val_custom_metrics = custom_metric_vad.compute()
            self._logger.info(f'Validation - Epoch {self.current_epoch} VAD: {val_custom_metrics}')
            val_custom_metrics = {f"vad_{k}": v for k, v in val_custom_metrics.items()}
            self._log_metrics_from_dict(val_custom_metrics, VAL)
            preds = custom_metric_vad.get_preds()
            labels = custom_metric_vad.get_labels()
            self.logger.experiment.add_pr_curve("Val/PR_Curve_vad", labels, preds,
                                                global_step=self.current_epoch)

        if self._custom_metrics_vap is not None:
            custom_metric_vap = self._custom_metrics_vap[VAL]
            val_custom_metrics = custom_metric_vap.compute()
            self._logger.info(f'Validation - Epoch {self.current_epoch} VAP: {val_custom_metrics}')
            val_custom_metrics = {f"vap_{k}": v for k, v in val_custom_metrics.items()}
            self._log_metrics_from_dict(val_custom_metrics, VAL)
            preds = custom_metric_vap.get_preds()
            labels = custom_metric_vap.get_labels()
            self.logger.experiment.add_pr_curve("Val/PR_Curve_vap", labels, preds,
                                                global_step=self.current_epoch)

        if self._turn_metrics_callback is not None:
            val_turn_metrics = self._turn_metrics_callback.metrics_epoch(VAL,
                                                                         epoch=self.current_epoch)
            self._log_metrics_from_dict(val_turn_metrics, VAL)

        avg_loss = self.trainer.callback_metrics.get("val_loss_vap_epoch")
        avg_loss_vad = self.trainer.callback_metrics.get("val_loss_vad_epoch")
        if avg_loss is not None:
            self._log_metric("vap", avg_loss, on_step=False, on_epoch=True)
        if avg_loss_vad is not None:
            self._log_metric("vad", avg_loss_vad, on_step=False, on_epoch=True)

        # Remove the raw step/epoch metric keys from the progress bar display
        for k in list(self.trainer.progress_bar_metrics.keys()):
            if k.startswith("val_"):
                del self.trainer.progress_bar_metrics[k]

    def on_train_epoch_end(self, *_):
        if self._custom_metrics_vad is not None:
            custom_metric_vad = self._custom_metrics_vad[TRAIN]
            train_custom_metrics = custom_metric_vad.compute()
            self._logger.info(f'Training - Epoch {self.current_epoch} VAD: {train_custom_metrics}')
            train_custom_metrics = {f"vad_{k}": v for k, v in train_custom_metrics.items()}
            self._log_metrics_from_dict(train_custom_metrics, TRAIN)
            preds = custom_metric_vad.get_preds()
            labels = custom_metric_vad.get_labels()
            self.logger.experiment.add_pr_curve("Train/PR_Curve_vad", labels, preds,
                                                global_step=self.current_epoch)

        if self._custom_metrics_vap is not None:
            custom_metric_vap = self._custom_metrics_vap[TRAIN]
            train_custom_metrics = custom_metric_vap.compute()
            self._logger.info(f'Training - Epoch {self.current_epoch} VAP: {train_custom_metrics}')
            train_custom_metrics = {f"vap_{k}": v for k, v in train_custom_metrics.items()}
            self._log_metrics_from_dict(train_custom_metrics, TRAIN)
            preds = custom_metric_vap.get_preds()
            labels = custom_metric_vap.get_labels()
            self.logger.experiment.add_pr_curve("Train/PR_Curve_vap", labels, preds,
                                                global_step=self.current_epoch)

        if self._turn_metrics_callback is not None:
            train_turn_metrics = self._turn_metrics_callback.metrics_epoch(TRAIN,
                                                                           epoch=self.current_epoch)
            self._log_metrics_from_dict(train_turn_metrics, TRAIN)

        avg_loss = self.trainer.callback_metrics.get("train_loss_vap_epoch")
        avg_loss_vad = self.trainer.callback_metrics.get("train_loss_vad_epoch")
        if avg_loss is not None:
            self._log_metric("vap", avg_loss, on_step=False, on_epoch=True)
        if avg_loss_vad is not None:
            self._log_metric("vad", avg_loss_vad, on_step=False, on_epoch=True)

        # Remove the raw step/epoch metric keys from the progress bar display
        for k in list(self.trainer.progress_bar_metrics.keys()):
            if k.startswith("train_"):
                del self.trainer.progress_bar_metrics[k]

        if self._turn_metrics_callback is not None:
            self._flush_panels(split=TRAIN)

        torch.cuda.empty_cache()
        if self._turn_metrics_callback is not None:
            self._turn_metrics_callback.reset()

        self._clear_step_metrics(self.trainer)
        if self._custom_metrics_vad is not None:
            for split in self._custom_metrics_vad.keys():
                self._custom_metrics_vad[split].reset()

        if self._custom_metrics_vap is not None:
            for split in self._custom_metrics_vap.keys():
                self._custom_metrics_vap[split].reset()

        for name, param in self.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.logger.experiment.add_histogram(name, param, global_step=self.current_epoch)

    def on_test_start(self):
        if self._turn_metrics_callback is not None:
            self._turn_metrics_callback.init_metrics(TEST, self.register_module)

    def test_step(self, batch, batch_idx, **kwargs):
        """test step"""
        out = self.shared_step(batch)

        out["loss_vap"] = self.objective.loss_vap(out["logits"], out["labels"], reduction="mean")
        out["loss_vad"] = self.objective.loss_vad(out["vad_logits"], out["vad_labels"])
        loss_vap = out["loss_vap"].detach()
        loss_vad = out["loss_vad"].detach()
        test_loss = out["loss_vap"] + out["loss_vad"]
        self._log_metric("loss_vap", loss_vap, on_step=True, on_epoch=True, batch_idx=batch_idx)
        self._log_metric("loss_vad", loss_vad, on_step=True, on_epoch=True, batch_idx=batch_idx)
        self._log_metric("loss", test_loss, on_step=True, on_epoch=True, batch_idx=batch_idx)

        # Metrics
        if self._custom_metrics_vad is not None:
            probs_vad = torch.sigmoid(out["vad_logits"])  # [B,T,2]
            preds_vad = (probs_vad > 0.5).int()
            n = out["vad_logits"].shape[-2]
            vad_labels = batch["vad"][:, :n, :].float()  # [B,T,2]
            self._custom_metrics_vad[TEST].update(preds_vad, vad_labels, loss_vad, probs_vad)

        if self._custom_metrics_vap is not None:
            probs_vap = torch.softmax(out["logits"], dim=-1)
            preds_vap = probs_vap.argmax(dim=-1)
            self._custom_metrics_vap[TEST].update(preds_vap, out["labels"], loss_vap, probs_vap)

        if self._turn_metrics_callback is not None:
            test_turntaking_preds, test_turntaking_targets = self._turn_metrics_callback.metrics_step(
                batch, out, split=TEST)

    def on_test_epoch_end(self, *_):
        if self._custom_metrics_vad is not None:
            custom_metric_vad = self._custom_metrics_vad[TEST]
            test_custom_metrics = custom_metric_vad.compute()
            self._logger.info(f'Test - Epoch {self.current_epoch} VAD: {test_custom_metrics}')
            test_custom_metrics = {f"vad_{k}": v for k, v in test_custom_metrics.items()}
            self._log_metrics_from_dict(test_custom_metrics, TEST)
            preds = custom_metric_vad.get_preds()
            labels = custom_metric_vad.get_labels()
            self.logger.experiment.add_pr_curve("Test/PR_Curve_vad", labels, preds,
                                                global_step=self.current_epoch)

        if self._custom_metrics_vap is not None:
            custom_metric_vap = self._custom_metrics_vap[TEST]
            test_custom_metrics = custom_metric_vap.compute()
            self._logger.info(f'Test - Epoch {self.current_epoch} VAP: {test_custom_metrics}')
            test_custom_metrics = {f"vap_{k}": v for k, v in test_custom_metrics.items()}
            self._log_metrics_from_dict(test_custom_metrics, TEST)
            preds = custom_metric_vap.get_preds()
            labels = custom_metric_vap.get_labels()
            self.logger.experiment.add_pr_curve("Test/PR_Curve_vap", labels, preds,
                                                global_step=self.current_epoch)

        if self._turn_metrics_callback is not None:
            test_turn_metrics = self._turn_metrics_callback.metrics_epoch(TEST,
                                                                          epoch=self.current_epoch)
            self._log_metrics_from_dict(test_turn_metrics, TEST)

        avg_loss = self.trainer.callback_metrics.get("test_loss_vap_epoch")
        avg_loss_vad = self.trainer.callback_metrics.get("test_loss_vad_epoch")
        if avg_loss is not None:
            self._log_metric("vap", avg_loss, on_step=False, on_epoch=True)
        if avg_loss_vad is not None:
            self._log_metric("vad", avg_loss_vad, on_step=False, on_epoch=True)

        # Remove the raw step/epoch metric keys from the progress bar display
        for k in list(self.trainer.progress_bar_metrics.keys()):
            if k.startswith("test_"):
                del self.trainer.progress_bar_metrics[k]

        if self._turn_metrics_callback is not None:
            self._flush_panels(split=TEST)

        torch.cuda.empty_cache()
        self._clear_step_metrics(self.trainer)
        if self._custom_metrics_vap is not None:
            for split in self._custom_metrics_vap.keys():
                self._custom_metrics_vap[split].reset()

    def _build_optimizer(self):
        """ Builds an optimizer based on the provided configuration"""
        opt_type = self.hyperparameters.optimizer.lower()
        weight_decay = self.hyperparameters.weight_decay

        if opt_type == "adam":
            opt = optim.Adam(self.parameters(), lr=self.lr, weight_decay=weight_decay)
        elif opt_type == "adamw":
            opt = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=weight_decay)
        elif opt_type == "sgd":
            opt = optim.SGD(
                self.parameters(),
                lr=self.lr,
                weight_decay=weight_decay,
                momentum=self.hyperparameters.momentum,
                nesterov=False,
                )
        elif opt_type == "rmsprop":
            opt = optim.RMSprop(
                self.parameters(),
                lr=self.lr,
                weight_decay=weight_decay,
                momentum=self.hyperparameters.momentum,
                alpha=self.hyperparameters.alpha,
                )
        elif opt_type == "adagrad":
            opt = optim.Adagrad(self.parameters(), lr=self.lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")

        self._logger.debug(f"Initialized optimizer: {opt_type.upper()} (lr={self.lr}, "
                           f"weight_decay={weight_decay})")
        return opt

    def _log_metric(self, base_name: str, value, on_step: bool = False, on_epoch: bool = True,
                    batch_idx: Optional[int] = None):
        """
        Logs metrics into:
        - `steps/{phase}/epoch_{X}/{metric}` → per-step curves (TensorBoard)
        - `{phase}/{metric}`                 → per-epoch trend (TensorBoard)
        - `{phase}_{metric}`                 → flat for callbacks & progress bar
        - `val_loss`                         → special raw key for EarlyStopping (only if base_name == "loss" and phase == "val")

        Args:
            base_name (str): Metric name like "loss", "f1", etc.
            value: Scalar or Tensor value.
            on_step (bool): Log every training step.
            on_epoch (bool): Log once per epoch.
            batch_idx (Optional[int]): Optional batch index for logging context.
        """
        if self.trainer.testing:
            phase = "test"
            logging_phase = "Testing"
        elif self.training:
            phase = "train"
            logging_phase = "Training"
        else:
            phase = "val"
            logging_phase = "Validation"

        epoch = getattr(self, "current_epoch", 0)

        # Step-level logging (visible in TensorBoard per-epoch folders)
        if on_step:
            self._logger.debug(f"{logging_phase} - epoch {epoch} step metric {batch_idx}: "
                               f"{phase}_{base_name} = {value:.4f}")
            self.log(
                f"steps/{phase}/epoch_{epoch}/{base_name}", value,
                on_step=True, on_epoch=False,
                logger=True, prog_bar=False, sync_dist=False
                )

        # Epoch-level logging (timeline view in TensorBoard)
        if on_epoch and not on_step:
            self._logger.info(f"{logging_phase} - epoch {epoch} {phase}_{base_name}: {value:.4f}")
            self.log(
                f"{phase}/{base_name}", value,
                on_step=False, on_epoch=True,
                logger=True, prog_bar=False, sync_dist=False
                )

        # Flat name for progress bar / callback compatibility
        self.log(
            f"{phase}_{base_name}", value,
            on_step=on_step, on_epoch=on_epoch,
            logger=False, prog_bar=True, sync_dist=False
            )

    def _log_metrics_from_dict(self, metrics: dict, split: str = "val"):
        """
        Logs a dictionary of metrics using a split prefix (e.g., 'train', 'val', or 'test').

        - Logs to TensorBoard as `split/MetricName`
        - Logs to progress bar as `split_MetricName` (not sent to TensorBoard)

        Args:
            metrics (dict): Metric name → value, e.g., {"F1": 0.82, "Acc": 0.9}
            split (str): "train", "val", or "test"
        """
        if not isinstance(metrics, dict):
            raise ValueError("metrics must be a dictionary")

        tb_metrics = {}
        bar_metrics = {}

        for k, v in metrics.items():
            name_clean = k.lower().replace(" ", "_")
            try:
                val = float(v)
            except Exception:
                continue

            tb_metrics[f"{split}/{name_clean}"] = val  # For TensorBoard (foldered)
            bar_metrics[f"{split}_{name_clean}"] = val  # For progress bar + callbacks

        self.log_dict(tb_metrics, prog_bar=False, logger=True, sync_dist=False)
        self.log_dict(bar_metrics, prog_bar=True, logger=False, sync_dist=False)

    def _clear_step_metrics(self, trainer):
        to_delete = [k for k in trainer.logged_metrics if k.startswith("steps/")]
        for k in to_delete:
            del trainer.logged_metrics[k]

    def _flush_panels(self, split: str = ""):
        panels = self._turn_metrics_callback.get_deferred_console_tables(split=split)

        if panels and self._logger.level >= logging.INFO:
            pb = next(
                (cb for cb in self.trainer.callbacks if isinstance(cb, ProgressBar)), None)
            if pb is not None and hasattr(pb, "_progress") and hasattr(pb._progress, "stop"):
                pb._progress.stop()
                console = pb._console
            else:
                console = Console()

            console.print("")  # New line for clarity
            if split == TRAIN:
                table_columns = Columns(panels)
            else:
                table_columns = panels

            console.print(table_columns)

            if self.logger is not None and hasattr(self.logger, "experiment"):
                buffer = StringIO()
                temp_console = Console(file=buffer, width=120)
                temp_console.print(table_columns)
                text_output = buffer.getvalue()
                self.logger.experiment.add_text(
                    f"Epoch {self.current_epoch} Turn-taking metrics",
                    f"```\n{text_output}\n```",
                    self.current_epoch)

            if pb is not None and hasattr(pb, "_progress") and hasattr(pb._progress, "start"):
                pb._progress.start()
