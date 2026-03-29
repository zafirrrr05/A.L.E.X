"""
src/training/pass_graph_dataset.py  (v2 — with quality filter)
───────────────────────────────────────────────────────────────
Builds a pass quality dataset where each sample is a PyG graph sequence.
Now filters out warmup/low-quality sequences using is_valid_sequence().

Filter stats (from data analysis):
    30,000 total → 26,195 valid match sequences → ~6,000+ pass samples
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.training.graph_dataset import (
    frame_to_player_graph,
    frame_to_team_graph,
)
from src.training.pass_utils import (
    detect_pass_events,
    extract_pass_window,
    pass_quality_label,
)
from src.utils.data_utils import is_valid_sequence

# ── normalisation — must match ssl_dataset.py ──
PITCH_W  = 1280.0
PITCH_H  = 768.0
VEL_CLIP = 5.0


def _normalise(team1, team2, ball):
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


def _window_to_graph_sequence(t1, t2, ball):
    W = t1.shape[0]
    sequence = []
    for f in range(W):
        pg = frame_to_player_graph(t1[f], t2[f], ball[f])
        tg = frame_to_team_graph(t1[f],  t2[f], ball[f])
        sequence.append((pg, tg))
    return sequence


class PassGraphDataset(Dataset):
    """
    Pass quality dataset using PyG graph sequences.
    Compatible with HierarchicalDualGATv2.

    Each sample:
        sequence : list of (player_graph, team_graph)
        label    : 1=good pass, 0=bad pass
        t_rel    : index of pass event in sequence (= pre)
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

        n_filtered       = 0
        n_events         = 0
        n_skip_bounds    = 0
        n_skip_label     = 0

        for fpath in tqdm(files, desc="Building pass graph dataset"):

            # ── quality filter ──
            if not is_valid_sequence(fpath):
                n_filtered += 1
                continue

            d = np.load(fpath)
            t1, t2, ball = _normalise(
                d["players_team1"],
                d["players_team2"],
                d["ball"],
            )

            t1_t   = torch.from_numpy(t1)
            t2_t   = torch.from_numpy(t2)
            ball_t = torch.from_numpy(ball)

            events = detect_pass_events(t1_t, t2_t, ball_t)
            n_events += len(events)

            for evt_t in events:
                w = extract_pass_window(t1_t, t2_t, ball_t,
                                        t=evt_t, pre=pre, post=post)
                if w is None:
                    n_skip_bounds += 1
                    continue

                y = pass_quality_label(t1_t, t2_t, ball_t, t_pass=evt_t)
                if y is None:
                    n_skip_label += 1
                    continue

                seq = _window_to_graph_sequence(
                    w["team1"].numpy(),
                    w["team2"].numpy(),
                    w["ball"].numpy(),
                )

                self.samples.append({
                    "sequence": seq,
                    "label":    y,
                    "t_rel":    pre,
                })

        pos = sum(s["label"] for s in self.samples)
        neg = len(self.samples) - pos

        print(f"PassGraphDataset: {len(files)} files  |  "
              f"filtered={n_filtered}  |  "
              f"{n_events} events  |  "
              f"{len(self.samples)} samples  |  "
              f"pos={pos} neg={neg}  |  "
              f"skipped: bounds={n_skip_bounds} label={n_skip_label}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        return {
            "sequence": s["sequence"],
            "label":    torch.tensor(s["label"], dtype=torch.long),
            "t_rel":    s["t_rel"],
        }