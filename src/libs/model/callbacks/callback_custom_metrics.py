# -*- coding: utf-8 -*-
import torch
from torchmetrics import Metric
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

from src.libs.logger.log import getLogger


class CCustomMetric(Metric):
    """
    Custom metric for binary VAD classification.
    Accumulates predictions and targets to compute final metrics at the end of validation/test.
    """
    def __init__(self, nan_threshold: float = 0.2, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self._logger = getLogger(self.__class__.__name__)

        self.nan_threshold = nan_threshold
        self.add_state("total_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("counter", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("accumulated_preds", default=[], dist_reduce_fx=None)
        self.add_state("accumulated_labels", default=[], dist_reduce_fx=None)
        self.add_state("accumulated_probs", default=[], dist_reduce_fx=None)

        self._m_batch_metrics_state = {}
        self._m_counter = 0

        self._m_val_loss_best_val = 1e10

        self._m_nan_rec_counter = 0
        self._m_nan_prec_counter = 0
        self._m_nan_tnr_counter = 0
        self._m_nan_npv_counter = 0
        self._m_nan_f1_counter = 0

        self._m_NAN_PRESENCE_THLD = 0.2
        self._m_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def update(self, preds: torch.Tensor, target: torch.Tensor, loss: torch.Tensor, probs: torch.Tensor):
        """
        Update metric state.

        Args:
            preds:  [B,T] for multiclass (VAP), [B,T,2] or [B,T] for binary (VAD).
            target: [B,T] (ints) or [B,T,2] (binary).
            loss:   scalar loss tensor.
            probs:  [B,T,C] for multiclass, [B,T,2] for binary multilabel.
        """
        self.total_loss += loss.detach().cpu()
        self.counter += 1

        # --- Multiclass case (VAP: 256 states) ---
        if probs.ndim == 3 and probs.shape[-1] == 256:
            self.accumulated_preds.append(preds.detach().cpu().view(-1))          # [N]
            self.accumulated_labels.append(target.detach().cpu().view(-1))        # [N]
            self.accumulated_probs.append(probs.detach().cpu().reshape(-1, probs.shape[-1]))  # [N,C]
        # --- Binary case (VAD: 2 speakers independent) ---
        elif probs.ndim == 3 and probs.shape[-1] == 2:
            # preds, target, probs are [B,T,2] → flatten to [N,2]
            self.accumulated_preds.append(preds.detach().cpu().reshape(-1, 2))
            self.accumulated_labels.append(target.detach().cpu().reshape(-1, 2))
            self.accumulated_probs.append(probs.detach().cpu().reshape(-1, 2))
        else:
            raise ValueError(f"Unexpected probs shape {probs.shape}")

    def compute(self) -> dict:
        """
        Compute average loss and metrics (F1, accuracy, precision, recall, AUC).
        Handles both VAD (binary multilabel) and VAP (multiclass).
        """
        preds = torch.cat(self.accumulated_preds)
        target = torch.cat(self.accumulated_labels)
        probs = torch.cat(self.accumulated_probs)
        avg_loss = self.total_loss.item() / max(self.counter.item(), 1)

        f1, accuracy, precision, recall, auc = 0.0, 0.0, 0.0, 0.0, 0.0
        try:
            preds_np = preds.numpy()
            target_np = target.numpy()
            probs_np = probs.numpy()
            if preds.ndim == 1:  # multiclass VAP
                pass
            elif preds.ndim == 2:  # binary multilabel VAD
                preds_np = preds_np.ravel()
                target_np = target_np.ravel()
                probs_np = probs_np.ravel()

            f1 = f1_score(target_np, preds_np, average="weighted")
            accuracy = accuracy_score(target_np, preds_np, normalize=True)
            precision = precision_score(target_np, preds_np, average="weighted")
            recall = recall_score(target_np, preds_np, average="weighted")
            auc = roc_auc_score(target_np, probs_np, multi_class="ovo", average="weighted").item()
        except Exception as e:
            self._logger.debug(f"Metric compute failed: {e}")

        return {
            "Avg Loss": round(avg_loss, 4),
            "F1": round(f1, 4),
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "AUC-ROC": round(auc, 4),
        }

    def reset(self):
        """
        Reset all metric state.
        """
        self.total_loss.zero_()
        self.counter.zero_()
        self.accumulated_preds.clear()
        self.accumulated_labels.clear()
        self.accumulated_probs.clear()

    def get_preds(self) -> torch.Tensor:
        """
        Get accumulated predictions for this metric instance.

        Returns:
            torch.Tensor: Flattened prediction tensor
        """
        if not self.accumulated_preds:
            return torch.tensor([])
        return torch.cat(self.accumulated_preds).view(-1)

    def get_labels(self) -> torch.Tensor:
        """
        Get accumulated labels for this metric instance.

        Returns:
            torch.Tensor: Flattened label tensor
        """
        if not self.accumulated_labels:
            return torch.tensor([])
        return torch.cat(self.accumulated_labels).view(-1)
