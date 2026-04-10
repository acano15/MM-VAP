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


class CBackchannel:
    """
    Backchannel region extraction from VAD. Operates on frames.

    Arguments:
        config: configuration dict or dataclass with:
        - post_onset_bc:       int, frames after speaker A starts to allow backchannel
        - pre_offset_bc:       int, frames before speaker A ends where backchannel may occur
        - min_gap_bc:          int, minimum silence duration to consider backchannel
        - max_duration_bc:     int, maximum length of a backchannel
        - metric_pad:          int, pad on backchannel onset used for evaluating
        - metric_dur:          int, duration of region used for evaluating
        min_silence_time: minimum silence time

    Speaker roles:
        - Speaker A: main speaker (floor holder)
        - Speaker B: listener (potential backchanneler)

    Active: "---"
    Silent: "..."

    # BACKCHANNEL (Insert during floor-holding)

    Timeline (Backchannel in pause or overlap):
                        A: ...---|-----------|.............
                        B: ------|...uh-huh..|-------------
                          BACKCHANNEL: |XXXX|

    Conditions:
        - Occurs when **Speaker A holds the floor**
        - Speaker B briefly becomes active
        - Often during **overlap** or **short silence**
        - Usually **short duration** (e.g., < 1s)
        - Can occur **while A is speaking**, or **immediately after a short pause**

    -----------------------------------------------------------
    # BACKCHANNEL NEGATIVE (No response when speaker holds floor)

                        A: ...---|-----------|.............
                        B: -------------------------------  # silent

        Used as training negative (no backchannel when expected)

    -----------------------------------------------------------
    # BACKCHANNEL METRICS WINDOW

                        A: ...---|-----------|.............
                        B: ------|...uh-huh..|-------------

        metric:                 |pad|---dur---|
        evaluation:             |...|XXXXXXXXX|

        - padding accounts for latency before the actual backchannel
        - duration defines the predicted/evaluated region

    -----------------------------------------------------------
    # Dialog states:
        Encoded using:
            0: Only A is speaking
            1: Silence
            2: Overlap
            3: Only B is speaking

    Typical transitions:

    BACKCHANNEL from silence:
        0 -> 1 -> 3       # A speaking, brief silence, B injects

    BACKCHANNEL in overlap:
        0 -> 2 -> 3       # A speaking, B overlaps briefly

    -----------------------------------------------------------
    # Notes:
        - Backchanneling often aligns with IPU ends or TRPs (transition relevance places)
        - Must respect timing windows — not every short overlap is a backchannel
        - Duration constraints (short) and speaker roles (listener vs speaker) are critical
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
        self.max_bc_time = config.bc_max_duration
        self.min_silence_time = min_silence_time
        self.min_context_time = config.min_context_time
        self.negatives_min_pad_left_time = config.bc_negative_pad_left_time
        self.negatives_min_pad_right_time = config.bc_negative_pad_right_time
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
        self.max_bc_frames = int(config.bc_max_duration * self.frame_hz)
        self.negatives_min_pad_left_frames = int(config.bc_negative_pad_left_time * self.frame_hz)
        self.negatives_min_pad_right_frames = int(config.bc_negative_pad_right_time * config.frame_hz)

    def __repr__(self) -> str:
        s = "Backhannel"
        s += "\n----------"
        s += f"\n  Time:"
        s += f"\n\tpre_cond_time              = {self.pre_cond_time}s"
        s += f"\n\tpost_cond_time             = {self.post_cond_time}s"
        s += f"\n\tmax_bc_time                = {self.max_bc_time}s"
        s += f"\n\tnegatives_left_pad_time    = {self.negatives_min_pad_left_time}s"
        s += f"\n\tnegatives_right_pad_time   = {self.negatives_min_pad_right_time}s"
        s += f"\n\tmin_context_time           = {self.min_context_time}s"
        s += f"\n\tmax_time                   = {self.max_time}s"
        s += f"\n  Frame:"
        s += f"\n\tpre_cond_frame             = {self.pre_cond_frames}"
        s += f"\n\tpost_cond_frame            = {self.post_cond_frames}"
        s += f"\n\tnegatives_left_pad_frames  = {self.negatives_min_pad_left_frames}"
        s += f"\n\tnegatives_right_pad_frames = {self.negatives_min_pad_right_frames}"
        s += f"\n\tprediction_region_frames   = {self.prediction_region_frames}"
        s += f"\n\tmax_bc_frames               = {self.max_bc_frames}"
        s += f"\n\tmin_context_frame          = {self.min_context_frames}"
        s += f"\n\tmax_frame                  = {self.max_frames}"
        return s

    def sample_negative_segment(self, region: Tuple[int, int, int]) -> Tuple[int, int, int]:
        region_start, region_end, speaker = region
        max_end = region_end - self.prediction_region_frames
        segment_start = random.randint(region_start, max_end)
        segment_end = segment_start + self.prediction_region_frames
        return segment_start, segment_end, speaker

    def __call__(self, vad: Tensor, ds: Optional[Tensor] = None, max_time: Optional[float] = None):
        batch_size = vad.shape[0]

        max_frame = self.max_frames
        if max_time is not None:
            max_frame = int(max_time * self.max_frames)

        if ds is None:
            ds = get_dialog_states(vad)

        self._logger.trace(f"Dialogue states shape: {ds.shape}")
        backchannel, pred_backchannel = [], []
        pred_backchannel_neg = []
        for b in range(batch_size):
            bc_samples = self.backchannel_regions(vad[b], ds=ds[b])

            bc_negative_regions = self.get_negative_sample_regions(vad=vad[b], ds=ds[b])
            self._logger.debug(f"Batch {b}: temporal regions: {bc_samples}")
            backchannel.append(bc_samples["backchannel"])
            pred_backchannel.append(bc_samples["pred_backchannel"])
            pred_backchannel_neg.append(bc_negative_regions)
        return {
            "backchannel": backchannel,
            "pred_backchannel": pred_backchannel,
            "pred_backchannel_neg": pred_backchannel_neg,
            }

    def backchannel_regions(self, vad: Tensor, ds: Tensor) -> Dict[str, List[Tuple[int, int, int]]]:
        assert vad.ndim == 2, f"expects vad of shape (n_frames, 2) but got {vad.shape}."

        filled_vad = fill_pauses(vad, ds)

        backchannel = []
        pred_backchannel = []
        for speaker in [0, 1]:
            start_of, duration_of, states = find_island_idx_len(filled_vad[..., speaker])
            if len(states) < 3:
                continue
            triads = states.unfold(0, size=3, step=1)
            steps = torch.where((triads == TRIAD_BC.to(triads.device).unsqueeze(0)).sum(-1) == 3)[0]
            if len(steps) == 0:
                continue
            for pre_silence in steps:
                bc = pre_silence + 1
                post_silence = pre_silence + 2
                ################################################
                # MINIMAL CONTEXT CONDITION
                ################################################
                if start_of[bc] < self.min_context_frames:
                    # print("Minimal context")
                    continue
                ################################################
                # MAXIMAL FRAME CONDITION
                ################################################
                if start_of[bc] >= self.max_frames:
                    # print("Max frame")
                    continue
                ################################################
                # MINIMAL DURATION CONDITION
                ################################################
                # Check bc duration
                if duration_of[bc] > self.max_bc_frames:
                    # print("Too Long")
                    continue
                ################################################
                # PRE CONDITION: No previous activity from bc-speaker
                ################################################
                if duration_of[pre_silence] < self.pre_cond_frames:
                    # print('not enough silence PRIOR to "bc"')
                    continue
                ################################################
                # POST CONDITION: No post activity from bc-speaker
                ################################################
                if duration_of[post_silence] < self.post_cond_frames:
                    # print('not enough silence POST to "bc"')
                    continue
                ################################################
                # ALL CONDITIONS MET
                ################################################
                # Is the other speakr active before this segment?
                backchannel.append(
                    (start_of[bc].item(), start_of[post_silence].item(), speaker)
                    )

                pred_bc_start = start_of[bc] - self.prediction_region_frames
                if pred_bc_start < self.min_context_frames:
                    continue

                pred_backchannel.append(
                    (pred_bc_start.item(), start_of[bc].item(), speaker)
                    )

        return {"backchannel": backchannel, "pred_backchannel": pred_backchannel}

    def get_negative_sample_regions(self, vad: Tensor, ds: Tensor) -> List[Tuple[int, int, int]]:
        min_dur_frames = self.negatives_min_pad_left_frames + self.negatives_min_pad_right_frames

        # fill pauses o recognize 'longer' segments of activity (including pauses)
        filled_vad = fill_pauses(vad, ds)
        ds_fill = get_dialog_states(filled_vad)
        index_of, duration_of, state_of = find_island_idx_len(ds_fill)

        neg_regions = []
        for current_speaker, current_speaker_state in enumerate([STATE_ONLY_A, STATE_ONLY_B]):
            next_potential_speaker = int(not current_speaker)
            dur = duration_of[state_of == current_speaker_state]
            idx = index_of[state_of == current_speaker_state]

            # iterate over all segments of longer activity
            for i, d in zip(idx, dur):
                ################################################
                # MINIMAL CONTEXT CONDITION
                ################################################
                # The total activity must allow for padding
                if d < min_dur_frames:
                    continue

                # START of region after `min_active_frames`
                start = (i + self.negatives_min_pad_left_frames).item()
                ################################################
                # CONTEXT (global/model) CONDITION
                ################################################
                # START of region must be after `min_context_frames`
                if start < self.min_context_frames:
                    start = self.min_context_frames

                # END of region prior to `min_pad_to_next_frames`
                end = (i + d - self.negatives_min_pad_right_frames).item()

                ################################################
                # MAXIMAL FRAME
                ################################################
                # end region can't span across last valid frame
                if end > self.max_frames:
                    end = self.max_frames

                ################################################
                # REGION SIZE
                ################################################
                # Is the final potential region larger than
                # the minimal required frames?
                # Also handles if end < start  (i.e. min_region_frames > 0)
                if end - start < self.prediction_region_frames:
                    continue

                neg_regions.append((start, end, next_potential_speaker))

        return neg_regions
