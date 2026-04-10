import torch
from torch import Tensor
from dataclasses import dataclass
import random
from typing import Dict, Optional, Tuple, List

from .events_config import CEventConfig
from .hold_shift import CHoldShift
from .backchannels import CBackchannel
from src.libs.utils import get_dialog_states
from src.libs.logger.log import getLogger

# Templates
TRIAD_SHIFT: Tensor = torch.tensor([[3, 1, 0], [0, 1, 3]])  # on Silence
TRIAD_SHIFT_OVERLAP: Tensor = torch.tensor([[3, 2, 0], [0, 2, 3]])
TRIAD_HOLD: Tensor = torch.tensor([[0, 1, 0], [3, 1, 3]])  # on silence
TRIAD_BC: Tensor = torch.tensor([0, 1, 0])

# Dialog states meaning
STATE_ONLY_A: int = 0
STATE_ONLY_B: int = 3
STATE_SILENCE: int = 1
STATE_BOTH: int = 2


class TurnTakingEvents:
    def __init__(self, conf: CEventConfig = None):
        self.conf = conf
        self._logger = getLogger(self.__class__.__name__)

        # Memory to add extra event in upcomming batches
        # if there is a discrepancy between
        # `pred_shift` & `pred_shift_neg` and
        # `pred_bc` & `pred_bc_neg` and
        self.add_extra = {"shift": 0, "pred_shift": 0, "pred_backchannel": 0}
        self.min_silence_time = conf.metric_time + conf.metric_pad_time

        assert (
            conf.min_context_time < conf.max_time
        ), "`minimum_context_time` must be lower than `max_time`"

        self.HS = CHoldShift(conf, min_silence_time=self.min_silence_time)
        self.BC = CBackchannel(conf, min_silence_time=self.min_silence_time)

    def __repr__(self) -> str:
        s = "TurnTakingEvents\n\n"
        s += self.BC.__repr__() + "\n"
        s += self.HS.__repr__()
        return s

    def get_total_ranges(self, a):
        return sum([len(events) for events in a])

    def sample_equal_amounts(
        self, n_to_sample, b_set, event_type, is_backchannel=False
        ):
        """Sample a subset from `b_set` of size of `n_to_sample`"""

        batch_size = len(b_set)

        # Create empty set
        subset = [[] for _ in range(batch_size)]

        # Flatten all events in B
        b_set_flat, batch_idx = [], []
        for b in range(batch_size):
            b_set_flat += b_set[b]
            batch_idx += [b] * len(b_set[b])

        # The maximum number of samples to sample
        n_max = len(b_set_flat)

        if n_max < n_to_sample:
            diff = n_to_sample - n_max
            self.add_extra[event_type] += diff
            n_to_sample = n_max
            self._logger.debug(f"Not enough events to sample from for {event_type}: adding "
                               f"{diff} to next batch")
        else:
            self._logger.debug(f"Sampling {n_to_sample} events from {n_max} available for {event_type}")
            diff = n_max - n_to_sample
            add_extra = min(diff, self.add_extra[event_type])
            n_to_sample += add_extra  # add extra 'negatives'
            # subtract the number of extra events we now sample
            self.add_extra[event_type] -= add_extra

        # Choose random a random subset from b_set
        for idx in random.sample(list(range(len(b_set_flat))), k=n_to_sample):
            self._logger.trace(f"Randomly selected index {idx} from {len(b_set_flat)}")
            b = batch_idx[idx]
            entry = b_set_flat[idx]
            if is_backchannel:
                entry = self.BC.sample_negative_segment(entry)
            subset[b].append(entry)
        return subset

    @torch.no_grad()
    def __call__(
        self, vad: Tensor, max_time: Optional[float] = None
        ) -> Dict[str, List[List[Tuple[int, int, int]]]]:
        assert (
            vad.ndim == 3
        ), f"Expects vad of shape (B, N_FRAMES, 2) but got {vad.shape}"
        ret = {}

        ds = get_dialog_states(vad)
        self._logger.trace(f"Dialogue states shape: {ds.shape}")
        bc = self.BC(vad, ds=ds, max_time=max_time)
        hs = self.HS(vad, ds=ds, max_time=max_time)
        ret.update(bc)
        ret.update(hs)

        self._logger.dev("Backchannel events: "
                         f"{ {k : self.get_total_ranges(v) for k, v in bc.items()}}")
        self._logger.dev(f"Hold/Shift events: "
                         f"{ {k : self.get_total_ranges(v) for k, v in hs.items()}}")

        # Sample equal amounts of "pre-hold" regions as "pre-shift"
        # ret["pred_shift_neg"] = self.sample_pred_shift_negatives(ret)
        n_pred_shift_negs_to_sample = self.get_total_ranges(ret["pred_shift"])
        self._logger.debug(f"Sampling {n_pred_shift_negs_to_sample} 'pred_shift_neg' events")
        # ret["pred_shift_neg"] = ret["pred_hold"]
        ret["pred_shift_neg"] = self.sample_equal_amounts(
            n_pred_shift_negs_to_sample, ret["pred_hold"], event_type="pred_shift")
        self._logger.dev(f"After sampling 'pred_shift_neg' events: {ret['pred_shift_neg']}")
        ret.pop("pred_hold")  # remove all pred_hold regions

        # Sample equal amounts of "pred_backchannel_neg" as "pred_backchannel"
        n_pred_bc_negs_to_sample = self.get_total_ranges(ret["pred_shift"])
        ret["pred_backchannel_neg"] = self.sample_equal_amounts(
            n_pred_bc_negs_to_sample, ret["pred_backchannel_neg"], event_type="pred_backchannel",
            is_backchannel=True)
        self._logger.dev(f"After sampling 'pred_backchannel_neg' events: {ret['pred_backchannel_neg']}")

        if self.conf.equal_hold_shift == 1:
            n_holds_to_sample = self.get_total_ranges(ret["shift"])
            ret["hold"] = self.sample_equal_amounts(n_holds_to_sample, ret["hold"],
                                                    event_type="shift")
            self._logger.dev(f"After sampling 'hold' events: {ret['hold']}")

        # renames
        ret["short"] = ret.pop("backchannel")
        return ret
