"""
src/training/pass_dataset.py  (v3)
───────────────────────────────────
Builds a labelled pass quality dataset from .npz sequences.

Key change vs v2:
  - Uses pass_quality_label() instead of pass_success()
  - Label is based on spatial geometry at the pass moment, not future frames
  - This gives much stronger signal for the model to learn from
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.training.pass_utils import (
    detect_pass_events,
    extract_pass_window,
    pass_quality_label,
)

# ── must match ssl_dataset.py ──
PITCH_W  = 1280.0
PITCH_H  = 768.0
VEL_CLIP = 5.0


def _normalise(team1: np.ndarray,
               team2: np.ndarray,
               ball:  np.ndarray):
    team1 = team1.copy().astype(np.float32)
    team2 = team2.copy().astype(np.float32)
    ball  = ball.copy().astype(np.float32)

    for arr in (team1, team2):
        arr[:, :, 0] /= PITCH_W
        arr[:, :, 1] /= PITCH_H
        arr[:, :, 2:] = np.clip(arr[:, :, 2:], -VEL_CLIP, VEL_CLIP) / VEL_CLIP

    ball[:, 0] /= PITCH_W
    ball[:, 1] /= PITCH_H
    ball[:, 2:] = np.clip(ball[:, 2:], -VEL_CLIP, VEL_CLIP) / VEL_CLIP

    return team1, team2, ball


class PassDataset(Dataset):
    """
    Scans .npz files, detects pass events, and labels each one
    using geometric pass quality (pressure + receiver space).

    Each sample:
        team1 : (pre+post, 11, 4)  normalised
        team2 : (pre+post, 11, 4)  normalised
        ball  : (pre+post, 4)      normalised
        label : int  1 = good pass, 0 = bad pass
    """

    def __init__(self, root_dir: str,
                 pre:  int = 5,
                 post: int = 10):

        self.samples = []

        files = sorted([
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith(".npz")
        ])

        n_events          = 0
        n_skipped_bounds  = 0
        n_skipped_label   = 0

        for fpath in tqdm(files, desc="Scanning sequences"):
            d = np.load(fpath)

            t1_raw   = d["players_team1"].astype(np.float32)
            t2_raw   = d["players_team2"].astype(np.float32)
            ball_raw = d["ball"].astype(np.float32)

            t1, t2, ball = _normalise(t1_raw, t2_raw, ball_raw)

            t1_t   = torch.from_numpy(t1)
            t2_t   = torch.from_numpy(t2)
            ball_t = torch.from_numpy(ball)

            events = detect_pass_events(t1_t, t2_t, ball_t)
            n_events += len(events)

            for evt_t in events:

                # extract window around the event
                w = extract_pass_window(t1_t, t2_t, ball_t,
                                        t=evt_t, pre=pre, post=post)
                if w is None:
                    n_skipped_bounds += 1
                    continue

                # geometric label AT the pass frame
                y = pass_quality_label(t1_t, t2_t, ball_t, t_pass=evt_t)
                if y is None:
                    n_skipped_label += 1
                    continue

                self.samples.append({
                    "team1": w["team1"],
                    "team2": w["team2"],
                    "ball":  w["ball"],
                    "label": y,
                })

        pos = sum(s["label"] for s in self.samples)
        neg = len(self.samples) - pos

        print(f"PassDataset: {len(files)} files  |  "
              f"{n_events} events detected  |  "
              f"{len(self.samples)} samples  |  "
              f"pos={pos} neg={neg}  |  "
              f"skipped: bounds={n_skipped_bounds} label={n_skipped_label}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        return {
            "team1": s["team1"].float(),
            "team2": s["team2"].float(),
            "ball":  s["ball"].float(),
            "label": torch.tensor(s["label"], dtype=torch.long),
        }