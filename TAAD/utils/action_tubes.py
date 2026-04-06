"""
Action Tube Construction — Section 3.5.

Converts the dense per-frame, per-player predictions into action tubes and
the three output formats required by Part 5 of the pipeline:

  1. JSON event dict  (API-2 format, Part 5 §5.10)
  2. Action tubes     [(start_frame, end_frame, class_idx, score), ...]
  3. Per-frame class-probability tensors  (raw, for downstream modules)

Step-by-step (Section 3.5):
  a. Per-player, per-frame argmax   →  raw label sequence
  b. Label smoothing / majority-vote over a sliding window
     (removes single-frame spurious detections)
  c. Run-length encode the smoothed sequence  →  action tube segments
  d. Filter by confidence threshold
  e. Format outputs
"""

from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch

from ..configs.labels  import IDX_TO_LABEL, API2_EVENT_TYPE, BACKGROUND
from ..configs.config  import CONF_THRESHOLD, IOU_THRESHOLD, SMOOTH_WINDOW


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ActionTube:
    """Represents a single detected action interval for one player."""
    player_idx  : int
    start_frame : int
    end_frame   : int
    class_idx   : int
    class_name  : str
    score       : float    # mean confidence over the tube


@dataclass
class EventDict:
    """
    API-2 output (Part 5 §5.10) — emitted to the Formation Module.

    Fields mirror the paper's event JSON (Section 5.3):
        start_time, end_time, team, player, event_type, confidence
    """
    start_frame : int
    end_frame   : int
    start_time  : float          # seconds
    end_time    : float          # seconds
    player_idx  : int
    team        : int            # 0 or 1  (unknown = -1 in proxy mode)
    event_type  : str            # canonical name from API2_EVENT_TYPE
    class_idx   : int
    confidence  : float


# ──────────────────────────────────────────────────────────────────────────────
# Label smoothing  (Section 3.5)
# ──────────────────────────────────────────────────────────────────────────────

def smooth_labels(
    labels: np.ndarray,   # (T,)  int
    window: int = SMOOTH_WINDOW,
) -> np.ndarray:
    """
    Majority-vote sliding window along the temporal axis.
    Removes isolated spurious label flips.
    """
    T = len(labels)
    smoothed = labels.copy()
    half = window // 2
    for t in range(T):
        lo = max(0, t - half)
        hi = min(T, t + half + 1)
        segment = labels[lo:hi]
        # Majority class in window
        counts = np.bincount(segment, minlength=int(labels.max()) + 2)
        smoothed[t] = int(counts.argmax())
    return smoothed


# ──────────────────────────────────────────────────────────────────────────────
# Run-length encoder  →  action tube segments
# ──────────────────────────────────────────────────────────────────────────────

def run_length_encode(
    labels: np.ndarray,   # (T,)  int
    scores: np.ndarray,   # (T,)  float  — max-class probability per frame
) -> list[tuple[int, int, int, float]]:
    """
    Returns list of (start, end, class_idx, mean_score) segments.
    Background (class 0) segments are excluded.
    """
    T = len(labels)
    segments = []
    if T == 0:
        return segments

    cur_cls   = labels[0]
    cur_start = 0
    cur_scores = [scores[0]]

    for t in range(1, T):
        if labels[t] != cur_cls:
            if cur_cls != BACKGROUND:
                segments.append((cur_start, t - 1, int(cur_cls),
                                  float(np.mean(cur_scores))))
            cur_cls   = labels[t]
            cur_start = t
            cur_scores = [scores[t]]
        else:
            cur_scores.append(scores[t])

    # Last segment
    if cur_cls != BACKGROUND:
        segments.append((cur_start, T - 1, int(cur_cls),
                          float(np.mean(cur_scores))))

    return segments


# ──────────────────────────────────────────────────────────────────────────────
# Main tube constructor
# ──────────────────────────────────────────────────────────────────────────────

def build_action_tubes(
    probs: torch.Tensor,                  # B×N×T×C  (softmax probabilities)
    team_ids: Optional[torch.Tensor],     # B×N×T   int  (may be all-zero in proxy mode)
    fps: float = 25.0,
    conf_threshold: float = CONF_THRESHOLD,
    smooth_window: int = SMOOTH_WINDOW,
    batch_idx: int = 0,
) -> dict:
    """
    Build all three output formats for one batch element.

    Args:
        probs          : B×N×T×C  — from model forward()
        team_ids       : B×N×T    — team membership per player per frame
        fps            : frames-per-second (for time conversion)
        conf_threshold : minimum mean tube confidence to keep
        smooth_window  : window size for label smoothing
        batch_idx      : which batch element to process

    Returns:
        dict with keys:
            "action_tubes"  : list[ActionTube]
            "event_dicts"   : list[EventDict]
            "frame_probs"   : np.ndarray  N×T×C  (per-frame probabilities)
    """
    p = probs[batch_idx].detach().cpu().numpy()   # N×T×C
    N, T, C = p.shape

    tid = (team_ids[batch_idx, :, 0].detach().cpu().numpy()
           if team_ids is not None else np.zeros(N, dtype=int))

    action_tubes: list[ActionTube] = []
    event_dicts:  list[EventDict]  = []

    for n in range(N):
        player_probs = p[n]                           # T×C
        raw_labels   = player_probs.argmax(axis=-1)   # (T,)
        max_scores   = player_probs.max(axis=-1)      # (T,)

        # Label smoothing
        smooth = smooth_labels(raw_labels, window=smooth_window)

        # Run-length encode → segments
        segs = run_length_encode(smooth, max_scores)

        for (sf, ef, cls_idx, score) in segs:
            if score < conf_threshold:
                continue

            tube = ActionTube(
                player_idx  = n,
                start_frame = sf,
                end_frame   = ef,
                class_idx   = cls_idx,
                class_name  = IDX_TO_LABEL.get(cls_idx, "unknown"),
                score       = score,
            )
            action_tubes.append(tube)

            ev = EventDict(
                start_frame = sf,
                end_frame   = ef,
                start_time  = sf / fps,
                end_time    = ef / fps,
                player_idx  = n,
                team        = int(tid[n]),
                event_type  = API2_EVENT_TYPE.get(cls_idx, "unknown"),
                class_idx   = cls_idx,
                confidence  = score,
            )
            event_dicts.append(ev)

    return {
        "action_tubes": action_tubes,
        "event_dicts":  event_dicts,
        "frame_probs":  p,           # N×T×C  (raw, for downstream modules)
    }


# ──────────────────────────────────────────────────────────────────────────────
# Serialisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def tubes_to_json(tubes: list[ActionTube]) -> str:
    return json.dumps([asdict(t) for t in tubes], indent=2)


def events_to_json(events: list[EventDict]) -> str:
    """
    Serialises event list to the API-2 JSON format described in Part 5 §5.10.
    """
    payload = []
    for e in events:
        payload.append({
            "start_time":  round(e.start_time, 3),
            "end_time":    round(e.end_time, 3),
            "start_frame": e.start_frame,
            "end_frame":   e.end_frame,
            "team":        e.team,
            "player":      e.player_idx,
            "event_type":  e.event_type,
            "confidence":  round(e.confidence, 4),
        })
    return json.dumps(payload, indent=2)
