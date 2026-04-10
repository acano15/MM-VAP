import torch
from torch import Tensor
from dataclasses import dataclass
import random
from typing import Dict, Optional, Tuple, List

from src.libs.utils import find_island_idx_len, fill_pauses, get_dialog_states
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


class CHoldShift:
    """
    Hold/Shift extraction from VAD. Operates of Frames.

    Arguments:
        config: configuration dict
        min_silence_time: minimum silence time

    Active: "---"
    Silent: "..."

    # SHIFTS
    onset:                                 |<-- only A -->|
    A:          ...........................|-------------------
    B:          ----------------|..............................
    offset:     |<--  only B -->|
    SHIFT:                      |XXXXXXXXXX|

    -----------------------------------------------------------
    # HOLDS
    onset:                                 |<-- only B -->|
    A:          ...............................................
    B:          ----------------|..........|-------------------
    offset:     |<--  only B -->|
    HOLD:                       |XXXXXXXXXX|

    -----------------------------------------------------------
    # NON-SHIFT
    Horizon:                        |<-- B majority -->|
    A:          .....................................|---------
    B:          ----------------|......|------|................
    non_shift:  |XXXXXXXXXXXXXXXXXXX|

    A future horizon window must contain 'majority' activity from
    the last speaker. In these moments we "know" a shift
    is a WRONG prediction. But closer to activity from the 'other'
    speaker, a turn-shift is appropriate.

    -----------------------------------------------------------
    # metrics
    e.g. shift

    onset:                                     |<-- only A -->|
    A:          ...............................|---------------
    B:          ----------------|..............................
    offset:     |<--  only B -->|
    SHIFT:                      |XXXXXXXXXXXXXX|
    metric:                     |...|XXXXXX|
    metric:                     |pad|  dur |
    -----------------------------------------------------------

    Using 'dialog states' consisting of 4 different states
    0. Only A is speaking
    1. Silence
    2. Overlap
    3. Only B is speaking

    Shift GAP:       0 -> 1 -> 3          3 -> 1 -> 0
    Shift Overlap:   0 -> 2 -> 3          3 -> 2 -> 0
    HOLD:            0 -> 1 -> 0          3 -> 1 -> 3
    """
    def __init__(self, config: dict, min_silence_time: float):
        self._logger = getLogger(self.__class__.__name__)

        self.prediction_region_on_active = config.sh_prediction_region_on_active

        # Time
        self.prediction_region_time = config.prediction_region_time
        self.long_onset_condition_time = config.long_onset_condition_time
        self.long_onset_region_time = config.long_onset_region_time
        self.pre_cond_time = config.bc_pre_cond_time
        self.post_cond_time = config.bc_post_cond_time
        self.min_silence_time = min_silence_time
        self.min_context_time = config.min_context_time
        self.max_time = config.max_time
        self.frame_hz = config.frame_hz

        # Frames
        self.pre_cond_frames = int(self.pre_cond_time * self.frame_hz)
        self.post_cond_frames = int(self.post_cond_time * self.frame_hz)
        self.prediction_region_frames = int(self.prediction_region_time * self.frame_hz)
        self.long_onset_condition_frames = int(self.long_onset_condition_time * self.frame_hz)
        self.long_onset_region_frames = int(self.long_onset_region_time * self.frame_hz)
        self.min_silence_frames = int(self.min_silence_time * self.frame_hz)
        self.min_context_frames = int(self.min_context_time * self.frame_hz)
        self.max_frames = int(self.max_time * self.frame_hz)

    def __repr__(self) -> str:
        s = "HoldShift"
        s += "\n---------"
        s += f"\n  Time:"
        s += f"\n\tpre_cond_time     = {self.pre_cond_time}s"
        s += f"\n\tpost_cond_time    = {self.post_cond_time}s"
        s += f"\n\tmin_silence_time  = {self.min_silence_time}s"
        s += f"\n\tmin_context_time  = {self.min_context_time}s"
        s += f"\n\tmax_time          = {self.max_time}s"
        s += f"\n  Frame:"
        s += f"\n\tpre_cond_frame    = {self.pre_cond_frames}"
        s += f"\n\tpost_cond_frame   = {self.post_cond_frames}"
        s += f"\n\tmin_silence_frame = {self.min_silence_frames}"
        s += f"\n\tmin_context_frame = {self.min_context_frames}"
        s += f"\n\tmax_frame         = {self.max_frames}"
        return s

    @torch.no_grad()
    def __call__(
        self,
        vad: Tensor,
        ds: Optional[Tensor] = None,
        max_time: Optional[float] = None,
        ) -> Dict[str, List[List[Tuple[int, int, int]]]]:
        assert (
            vad.ndim == 3
        ), f"Expected vad.ndim=3 (B, N_FRAMES, 2) but got {vad.shape}"

        max_frame = self.max_frames
        if max_time is not None:
            max_frame = int(max_time * self.frame_hz)

        batch_size = vad.shape[0]

        if ds is None:
            ds = get_dialog_states(vad)

        self._logger.trace(f"Dialogue states shape: {ds.shape}")
        shift, hold, long = [], [], []
        pred_shift, pred_hold = [], []
        for b in range(batch_size):
            tmp_sh = self.hold_shift_regions(vad=vad[b], ds=ds[b])
            self._logger.debug(f"Batch {b}: temporal regions: {tmp_sh}")
            shift.append(tmp_sh["shift"])
            hold.append(tmp_sh["hold"])
            long.append(tmp_sh["long"])
            pred_shift.append(tmp_sh["pred_shift"])
            pred_hold.append(tmp_sh["pred_hold"])
        return {
            "shift": shift,
            "hold": hold,
            "long": long,
            "pred_shift": pred_shift,
            "pred_hold": pred_hold,
            }

    def hold_shift_regions(self, vad: Tensor, ds: Tensor) -> Dict[str, List[Tuple[int, int, int]]]:
        assert vad.ndim == 2, f"expects vad of shape (n_frames, 2) but got {vad.shape}."

        start_of, duration_of, states = find_island_idx_len(ds)
        filled_vad = fill_pauses(vad, ds, islands=(start_of, duration_of, states))

        # If we have less than 3 unique dialog states
        # then we have no valid transitions
        if len(states) < 3:
            return {"shift": [], "hold": [], "long": [], "pred_shift": [], "pred_hold": []}

        triads = states.unfold(0, size=3, step=1)

        # SHIFTS
        shifts, pred_shifts, long_onset = self.get_hs_regions(
            triads=triads,
            filled_vad=filled_vad,
            triad_label=TRIAD_SHIFT.to(vad.device),
            start_of=start_of,
            duration_of=duration_of)

        # HOLDS
        holds, pred_holds, _ = self.get_hs_regions(
            triads=triads,
            filled_vad=filled_vad,
            triad_label=TRIAD_HOLD.to(vad.device),
            start_of=start_of,
            duration_of=duration_of)

        return {
            "shift": shifts,
            "hold": holds,
            "long": long_onset,
            "pred_shift": pred_shifts,
            "pred_hold": pred_holds,
            }

    def get_hs_regions(self, triads: Tensor, filled_vad: Tensor, triad_label: Tensor, start_of: Tensor,
                       duration_of: Tensor) -> Tuple[
                       List[Tuple[int, int, int]],
                       List[Tuple[int, int, int]],
                       List[Tuple[int, int, int]]]:
        """
        get regions defined by `triad_label`
        """

        region = []
        prediction_region = []
        long_onset_region = []

        # check if label is hold or shift
        # if the same speaker continues after silence -> hold
        hold_cond = triad_label[0, 0] == triad_label[0, -1]
        next_speakers, steps = torch.where(
            (triads == triad_label.unsqueeze(1)).sum(-1) == 3
            )
        # No matches -> return
        if len(next_speakers) == 0:
            return [], [], []

        for last_onset, next_speaker in zip(steps, next_speakers):
            not_next_speaker = int(not next_speaker)
            prev_speaker = next_speaker if hold_cond else not_next_speaker
            not_prev_speaker = 0 if prev_speaker == 1 else 1
            # All shift triads e.g. [3, 1, 0] centers on the silence segment
            # If we find a triad match at step 's' then the actual SILENCE segment
            # STARTS:           on the next step -> add 1
            # ENDS/next-onset:  on the two next step -> add 2
            silence = last_onset + 1
            next_onset = last_onset + 2
            ################################################
            # MINIMAL CONTEXT CONDITION
            ################################################
            if start_of[silence] < self.min_context_frames:
                continue
            ################################################
            # MAXIMAL FRAME CONDITION
            ################################################
            if start_of[silence] >= self.max_frames:
                continue
            ################################################
            # MINIMAL SILENCE CONDITION
            ################################################
            # Check silence duration
            if duration_of[silence] < self.min_silence_frames:
                continue
            ################################################
            # PRE CONDITION: ONLY A SINGLE PREVIOUS SPEAKER
            ################################################
            # Check `pre_cond_frames` before start of silence
            # to make sure only a single speaker was active
            sil_start = start_of[silence]
            pre_start = sil_start - self.pre_cond_frames
            pre_start = pre_start if pre_start > 0 else 0
            correct_is_active = (
                filled_vad[pre_start:sil_start, prev_speaker].sum() == self.pre_cond_frames
            )
            if not correct_is_active:
                continue
            other_is_silent = filled_vad[pre_start:sil_start, not_prev_speaker].sum() == 0
            if not other_is_silent:
                continue
            ################################################
            # POST CONDITION: ONLY A SINGLE PREVIOUS SPEAKER
            ################################################
            # Check `post_cond_frames` after start of onset
            # to make sure only a single speaker is to be active
            onset_start = start_of[next_onset]
            onset_region_end = onset_start + self.post_cond_frames
            correct_is_active = (
                filled_vad[onset_start:onset_region_end, next_speaker].sum()
                == self.post_cond_frames
            )
            if not correct_is_active:
                continue
            other_is_silent = (
                filled_vad[onset_start:onset_region_end, not_next_speaker].sum() == 0
            )
            if not other_is_silent:
                continue
            ################################################
            # ALL CONDITIONS MET
            ################################################
            region.append((sil_start.item(), onset_start.item(), next_speaker.item()))

            ################################################
            # LONG ONSET CONDITION
            ################################################
            # if we have a valid shift we check if the onset
            # of the next segment is longer than `long_onset_condition_frames`
            # and if true we add the region
            if not hold_cond and duration_of[next_onset] >= self.long_onset_condition_frames:
                # We add the 'long-onset' region defined by `long_onset_region_frames`
                # the condition is used to define "yea, this is an onset of a 'long' region"
                # whereas the `long_onset_region_frames` define the area in which we wish
                # to make predictions with the model.
                long_onset_region.append(
                    (
                        onset_start.item(),
                        (onset_start + self.long_onset_region_frames).item(),
                        next_speaker.item(),
                        )
                    )

            ################################################
            # PREDICTION REGION CONDITION
            ################################################
            # The prediction region is defined at the end of the previous
            # activity, not inside the silences.

            # IF PREDICTION_REGION_ON_ACTIVE = FALSE
            # We don't care about the previous activity but only take
            # `prediction_region_frames` prior to the relevant hold/shift silence.
            # e.g. if prediction_region_frames=100 and the last segment prior to the
            # relevant hold/shift silence was 70 frames the prediction region would include
            # < 30 frames of silence (a pause or a shift (could be a quick back and forth limited by
            # the condition variables...))
            if self.prediction_region_on_active:
                # We make sure that the last VAD segments
                # of the last speaker is longer than
                # `prediction_region_frames`
                if duration_of[last_onset] < self.prediction_region_frames:
                    continue

            # that if the last activity
            prediction_start = sil_start - self.prediction_region_frames

            ################################################
            # MINIMAL CONTEXT CONDITION (PREDICTION)
            ################################################
            if prediction_start < self.min_context_frames:
                continue

            prediction_region.append(
                (prediction_start.item(), sil_start.item(), next_speaker.item())
                )

        return region, prediction_region, long_onset_region
