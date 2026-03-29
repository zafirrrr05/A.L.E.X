"""
src/training/formation_analysis.py  (v2 — with quality filter)
───────────────────────────────────────────────────────────────
Detects football formations from tracking data.

Changes vs v1:
  - is_valid_sequence() filter added to FormationDataset
  - Team1-only labels (team2 has too few players in many sequences)
  - estimate_opponent_formation() added for team2 inference
  - k=3 forced (football always has 3 outfield lines)
  - Clusters on relative x (removes camera/FOV bias)
  - Broader fuzzy formation mapping
"""

import os
import numpy as np
import torch
from sklearn.cluster import KMeans
from tqdm import tqdm

from src.utils.data_utils import is_valid_sequence

# ── formation label map ──
FORMATION_CLASSES = {
    "4-4-2":   0,
    "4-3-3":   1,
    "4-2-3-1": 2,
    "3-5-2":   3,
    "other":   4,
}
FORMATION_NAMES = {v: k for k, v in FORMATION_CLASSES.items()}

PITCH_W = 1280.0
PITCH_H = 768.0

COUNTER_FORMATIONS = {
    "4-3-3":   ["4-4-2", "4-2-3-1"],
    "4-4-2":   ["4-3-3", "4-2-3-1"],
    "4-2-3-1": ["4-4-2", "4-3-3"],
    "3-5-2":   ["4-3-3", "4-4-2"],
    "other":   ["other"],
}


# ─────────────────────────────────────────────
#  CORE FORMATION DETECTOR
# ─────────────────────────────────────────────

def detect_formation(team_raw: np.ndarray,
                     attacking_direction: str = "right") -> dict:
    """
    Detect formation from a (T, 11, 4) player array.

    team_raw            : (T, 11, 4)  raw pixel coords
    attacking_direction : "right" or "left"

    Returns dict with formation_str, formation_class, line_counts etc.
    """
    # normalise positions
    pos = team_raw[:, :, :2].copy().astype(np.float32)
    pos[:, :, 0] /= PITCH_W
    pos[:, :, 1] /= PITCH_H

    # valid players — present across all frames
    valid = ~np.all(team_raw == 0, axis=(0, 2))   # (11,)
    mean_pos  = pos.mean(axis=0)                   # (11, 2)
    valid_idx = np.where(valid)[0]

    if len(valid_idx) < 4:
        return _empty_result(mean_pos)

    # ── detect goalkeeper ──
    valid_x = mean_pos[valid_idx, 0]
    if attacking_direction == "right":
        gk_local = np.argmin(valid_x)
    else:
        gk_local = np.argmax(valid_x)

    gk_idx   = valid_idx[gk_local]
    outfield = np.delete(valid_idx, gk_local)

    if len(outfield) < 3:
        return _empty_result(mean_pos, gk_idx=int(gk_idx))

    # ── cluster on x relative to team centroid (removes camera bias) ──
    outfield_x_abs = mean_pos[outfield, 0]
    team_mean_x    = outfield_x_abs.mean()
    outfield_x_rel = (outfield_x_abs - team_mean_x).reshape(-1, 1)

    best_k, best_lines = _cluster_into_lines(
        outfield_x_rel, outfield, attacking_direction
    )

    line_counts   = [len(l) for l in best_lines]
    formation_str = _counts_to_formation(line_counts)

    return {
        "formation_str":   formation_str,
        "formation_class": FORMATION_CLASSES.get(formation_str,
                                                  FORMATION_CLASSES["other"]),
        "line_counts":     sorted(line_counts, reverse=True),
        "mean_positions":  mean_pos,
        "gk_idx":          int(gk_idx),
        "lines":           best_lines,
        "confidence":      "detected",
        "valid_players":   int(valid.sum()),
    }


def _empty_result(mean_pos, gk_idx=-1):
    return {
        "formation_str":   "other",
        "formation_class": FORMATION_CLASSES["other"],
        "line_counts":     [],
        "mean_positions":  mean_pos,
        "gk_idx":          gk_idx,
        "lines":           [],
        "confidence":      "detected",
        "valid_players":   0,
    }


def _cluster_into_lines(outfield_x_rel, outfield_idx,
                         attacking_direction, k=3):
    """Force k=3 clusters — football always has 3 outfield lines."""
    k = min(k, len(outfield_idx))
    if k < 2:
        return 1, [outfield_idx]

    km     = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(outfield_x_rel)

    # sort lines from defensive to attacking by mean x
    lines      = [outfield_idx[labels == c] for c in range(k)]
    line_means = [outfield_x_rel[labels == c].mean() for c in range(k)]
    order      = np.argsort(line_means)
    if attacking_direction == "left":
        order = order[::-1]

    lines = [lines[i] for i in order]
    return k, lines


def _counts_to_formation(counts: list) -> str:
    if not counts:
        return "other"

    c = tuple(sorted(counts, reverse=True))

    if len(c) == 3:
        a, b, cc = c
        if (a, b, cc) == (4, 4, 2): return "4-4-2"
        if (a, b, cc) == (4, 3, 3): return "4-3-3"
        if (a, b, cc) == (5, 3, 2): return "3-5-2"
        if (a, b, cc) == (5, 2, 3): return "3-5-2"
        if a == 4 and b == 4:       return "4-4-2"
        if a == 4 and b == 3:       return "4-3-3"
        if a == 3 and b == 4:       return "4-3-3"
        if a == 5 and b >= 3:       return "3-5-2"
        if a == 4 and b == 2:       return "4-2-3-1"
        if a >= 4 and b >= 3:       return "4-3-3"
        return "other"

    return "other"


# ─────────────────────────────────────────────
#  OPPONENT FORMATION ESTIMATION
# ─────────────────────────────────────────────

def estimate_opponent_formation(known_formation: str,
                                team2_raw: np.ndarray) -> dict:
    """
    Estimate team2 formation.
    Uses direct detection if 8+ valid players, otherwise infers
    from team1's formation using common counter-formations.
    """
    valid = int((~np.all(team2_raw == 0, axis=(0, 2))).sum())

    if valid >= 8:
        result = detect_formation(team2_raw, attacking_direction="left")
        result["confidence"] = "detected"
        return result

    likely = COUNTER_FORMATIONS.get(known_formation, ["other"])
    return {
        "formation_str":   likely[0],
        "formation_class": FORMATION_CLASSES.get(likely[0], 4),
        "line_counts":     [],
        "mean_positions":  None,
        "gk_idx":          -1,
        "lines":           [],
        "confidence":      "estimated",
        "valid_players":   valid,
    }


# ─────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────

class FormationDataset(torch.utils.data.Dataset):
    """
    Builds formation labels from .npz files.
    Team1 labels only — team2 has insufficient player coverage.
    Filters warmup/low-quality sequences automatically.

    Each sample:
        team1    : (T, 11, 4) normalised
        team2    : (T, 11, 4) normalised
        ball     : (T, 4)     normalised
        label_t1 : int  formation class for team1
    """

    def __init__(self, root_dir: str):

        self.samples = []

        files = sorted([
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith(".npz")
        ])

        n_filtered    = 0
        label_counts  = {k: 0 for k in FORMATION_CLASSES.values()}

        for fpath in tqdm(files, desc="Building formation dataset"):

            # ── quality filter ──
            if not is_valid_sequence(fpath):
                n_filtered += 1
                continue

            d = np.load(fpath)

            t1_raw   = d["players_team1"].astype(np.float32)
            t2_raw   = d["players_team2"].astype(np.float32)
            ball_raw = d["ball"].astype(np.float32)

            # detect team1 formation (reliable)
            f1 = detect_formation(t1_raw, attacking_direction="right")

            # skip if team1 formation is "other" — poor quality clip
            if f1["formation_str"] == "other":
                continue

            # normalise
            t1   = t1_raw.copy()
            t2   = t2_raw.copy()
            ball = ball_raw.copy()

            for arr in (t1, t2):
                arr[:, :, 0] /= PITCH_W
                arr[:, :, 1] /= PITCH_H

            ball[:, 0] /= PITCH_W
            ball[:, 1] /= PITCH_H

            label_counts[f1["formation_class"]] += 1

            self.samples.append({
                "team1":    torch.from_numpy(t1),
                "team2":    torch.from_numpy(t2),
                "ball":     torch.from_numpy(ball),
                "label_t1": f1["formation_class"],
            })

        print(f"FormationDataset: {len(self.samples)} clips  "
              f"(filtered={n_filtered})")
        print("Formation distribution (team1):")
        for cls, count in label_counts.items():
            print(f"  {FORMATION_NAMES[cls]:12s}: {count:6d}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        return {
            "team1":    s["team1"].float(),
            "team2":    s["team2"].float(),
            "ball":     s["ball"].float(),
            "label_t1": torch.tensor(s["label_t1"], dtype=torch.long),
        }


# ─────────────────────────────────────────────
#  SINGLE-CLIP INFERENCE
# ─────────────────────────────────────────────

def formation_from_npz(npz_path: str) -> dict:
    """Formation detection for a single .npz file — team1 only."""
    d  = np.load(npz_path)
    t1 = d["players_team1"].astype(np.float32)
    return {"team1": detect_formation(t1, attacking_direction="right")}


def formation_from_npz_both_teams(npz_path: str) -> dict:
    """Formation for both teams — team2 estimated if insufficient players."""
    d  = np.load(npz_path)
    t1 = d["players_team1"].astype(np.float32)
    t2 = d["players_team2"].astype(np.float32)
    f1 = detect_formation(t1, attacking_direction="right")
    f2 = estimate_opponent_formation(f1["formation_str"], t2)
    return {"team1": f1, "team2": f2}