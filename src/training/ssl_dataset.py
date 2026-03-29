import os
from cv2 import data
import numpy as np
import torch
from torch.utils.data import Dataset


class TeamSequenceDataset(Dataset):

    def __init__(self, root_dir, min_len=30):
        self.root_dir = root_dir
        self.files = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith(".npz")
        ]

        self.min_len = min_len

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        data = np.load(self.files[idx])

        team1 = data["players_team1"].astype(np.float32)   # (T,11,4)
        team2 = data["players_team2"].astype(np.float32)   # (T,11,4)
        ball  = data["ball"].astype(np.float32)            # (T,4)

        # ── NORMALISATION LINES ──
        # confirm your actual pitch coordinate range first (see note below)
        PITCH_W = 1280.0   # your frame width  (1147 max → use 1280 as safe ceiling)
        PITCH_H = 768.0    # your frame height (750 max  → use 768  as safe ceiling)

        # positions (cx, cy) → divide by pitch size
        team1[:, :, 0] /= PITCH_W   # cx
        team1[:, :, 1] /= PITCH_H   # cy
        team2[:, :, 0] /= PITCH_W
        team2[:, :, 1] /= PITCH_H
        ball[:, 0] /= PITCH_W
        ball[:, 1] /= PITCH_H

        # velocities (vx, vy) → clip and scale
        team1[:, :, 2:] = np.clip(team1[:, :, 2:], -5, 5) / 5.0
        team2[:, :, 2:] = np.clip(team2[:, :, 2:], -5, 5) / 5.0
        ball[:, 2:]     = np.clip(ball[:, 2:],     -5, 5) / 5.0
        # ── END NORMALISATION ──    

        if team1.shape[0] < self.min_len:
            raise RuntimeError("Sequence too short")

        return {
            "team1": torch.from_numpy(team1),
            "team2": torch.from_numpy(team2),
            "ball": torch.from_numpy(ball)
        }
    

def sample_future_prediction(seq, past_len=20, future_offset=10):

    team1 = seq["team1"]
    team2 = seq["team2"]
    ball  = seq["ball"]

    T = team1.shape[0]

    max_start = T - past_len - future_offset
    if max_start <= 0:
        raise RuntimeError("Sequence too short for future prediction task")

    t0 = torch.randint(0, max_start, (1,)).item()

    return {
        "past_team1": team1[t0:t0+past_len],
        "past_team2": team2[t0:t0+past_len],
        "past_ball":  ball[t0:t0+past_len],

        "target_team1": team1[t0+past_len+future_offset-1],
        "target_team2": team2[t0+past_len+future_offset-1],
        "target_ball":  ball[t0+past_len+future_offset-1],
    }


def sample_masked_players(seq, mask_ratio=0.2):
    """
    Dual team masked player modeling.

    Returns:
        masked_team1 : [T,11,4]
        masked_team2 : [T,11,4]
        target_team1 : [T,11,4]
        target_team2 : [T,11,4]
        mask         : [22]  (players to reconstruct in the LAST frame)
    """

    team1 = seq["team1"].clone()
    team2 = seq["team2"].clone()

    T = team1.shape[0]

    # 22 player slots
    mask = torch.rand(22) < mask_ratio

    masked_t1 = team1.clone()
    masked_t2 = team2.clone()

    # first 11 → team1
    mask_t1 = mask[:11]
    mask_t2 = mask[11:]

    masked_t1[:, mask_t1, :] = 0.0
    masked_t2[:, mask_t2, :] = 0.0

    return {
        "masked_team1": masked_t1,
        "masked_team2": masked_t2,
        "target_team1": team1,
        "target_team2": team2,
        "mask": mask
    }


def compute_possession(team1, team2, ball):

    # team1, team2 : [11,4]
    # ball : [4]

    all_players = torch.cat([team1, team2], dim=0)  # [22,4]

    d = torch.norm(all_players[:, :2] - ball[:2], dim=1)

    return torch.argmin(d)


def sample_possession(seq, horizon=10):

    team1 = seq["team1"]
    team2 = seq["team2"]
    ball  = seq["ball"]

    T = team1.shape[0]

    t0 = torch.randint(0, T-horizon-1, (1,)).item()

    p0 = compute_possession(team1[t0], team2[t0], ball[t0])
    p1 = compute_possession(team1[t0+horizon], team2[t0+horizon], ball[t0+horizon])

    same = int(p0 == p1)

    return {
        "team1": team1[t0],
        "team2": team2[t0],
        "ball":  ball[t0],
        "label": torch.tensor(same, dtype=torch.long)
    }

def sample_temporal_order(seq, clip_len=10):

    team1 = seq["team1"]
    team2 = seq["team2"]
    ball  = seq["ball"]

    T = team1.shape[0]

    t0 = torch.randint(0, T - 2*clip_len, (1,)).item()

    c1_t1 = team1[t0:t0+clip_len]
    c1_t2 = team2[t0:t0+clip_len]
    c1_b  = ball[t0:t0+clip_len]

    c2_t1 = team1[t0+clip_len:t0+2*clip_len]
    c2_t2 = team2[t0+clip_len:t0+2*clip_len]
    c2_b  = ball[t0+clip_len:t0+2*clip_len]

    if torch.rand(1).item() < 0.5:
        label = 1
        return c1_t1, c1_t2, c2_t1, c2_t2, c1_b, c2_b, label
    else:
        label = 0
        return c2_t1, c2_t2, c1_t1, c1_t2, c2_b, c1_b, label
    

def augment_view(team1, team2, ball):

    noisy1 = team1.clone()
    noisy2 = team2.clone()

    noisy1[:, :, :2] += 0.01 * torch.randn_like(noisy1[:, :, :2])
    noisy2[:, :, :2] += 0.01 * torch.randn_like(noisy2[:, :, :2])

    drop1 = torch.rand(team1.shape[1]) < 0.2
    drop2 = torch.rand(team2.shape[1]) < 0.2

    noisy1[:, drop1, :] = 0.0
    noisy2[:, drop2, :] = 0.0

    return noisy1, noisy2, ball


def sample_contrastive(seq):

    team1 = seq["team1"]
    team2 = seq["team2"]
    ball  = seq["ball"]

    v1_t1, v1_t2, v1_b = augment_view(team1, team2, ball)
    v2_t1, v2_t2, v2_b = augment_view(team1, team2, ball)

    return {
        "view1_team1": v1_t1,
        "view1_team2": v1_t2,
        "view1_ball":  v1_b,
        "view2_team1": v2_t1,
        "view2_team2": v2_t2,
        "view2_ball":  v2_b,
    }