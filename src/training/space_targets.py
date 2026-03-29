"""
src/training/space_targets.py  (v2)
─────────────────────────────────────
Builds per-frame space control targets from tracking data.

Key fixes vs v1:
  - Uses BOTH teams (22 players) for contested space calculation
  - Normalisation is done inside this module — caller does not need to normalise
  - Added build_space_targets_from_npz() for direct .npz loading
  - Grid default changed to match pixel-normalised coords
"""

import torch
import numpy as np

# ── must match pass_dataset.py and ssl_dataset.py ──
PITCH_W = 1280.0
PITCH_H = 768.0


def make_grid(H: int = 24, W: int = 16) -> torch.Tensor:
    """
    Build a (H, W, 2) grid of normalised pitch coordinates.
    H rows (y-axis), W cols (x-axis), values in [0, 1].
    """
    xs = torch.linspace(0, 1, W)
    ys = torch.linspace(0, 1, H)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1)   # (H, W, 2)


def build_space_targets(team1:   torch.Tensor,
                        team2:   torch.Tensor,
                        ball:    torch.Tensor,
                        grid_xy: torch.Tensor,
                        already_normalised: bool = False) -> torch.Tensor:
    """
    Compute per-frame space control map.

    team1   : (T, 11, 4)  raw or normalised
    team2   : (T, 11, 4)
    ball    : (T, 4)
    grid_xy : (H, W, 2)   normalised [0,1] coords — use make_grid()

    Returns : (T, H, W)   float in [0,1]
              high value = team1 controls this space
              low value  = team2 controls this space
    """
    if not already_normalised:
        team1 = team1.clone()
        team2 = team2.clone()
        ball  = ball.clone()

        team1[:, :, 0] /= PITCH_W
        team1[:, :, 1] /= PITCH_H
        team2[:, :, 0] /= PITCH_W
        team2[:, :, 1] /= PITCH_H
        ball[:, 0] /= PITCH_W
        ball[:, 1] /= PITCH_H

    T = team1.shape[0]
    H, W = grid_xy.shape[:2]
    targets = torch.zeros(T, H, W)

    grid_flat = grid_xy.view(-1, 2)   # (H*W, 2)

    for t in range(T):

        # valid player masks (skip zero-padded rows)
        valid1 = ~torch.all(team1[t] == 0, dim=-1)   # (11,)
        valid2 = ~torch.all(team2[t] == 0, dim=-1)   # (11,)

        p1 = team1[t][valid1, :2]   # (V1, 2)
        p2 = team2[t][valid2, :2]   # (V2, 2)
        b  = ball[t, :2]            # (2,)

        # distance from every grid cell to nearest team1 player
        if len(p1) > 0:
            diff1 = grid_flat[:, None, :] - p1[None, :, :]   # (G, V1, 2)
            d1 = torch.norm(diff1, dim=-1).min(dim=1)[0]      # (G,)
        else:
            d1 = torch.full((H * W,), 1e9)

        # distance from every grid cell to nearest team2 player
        if len(p2) > 0:
            diff2 = grid_flat[:, None, :] - p2[None, :, :]   # (G, V2, 2)
            d2 = torch.norm(diff2, dim=-1).min(dim=1)[0]      # (G,)
        else:
            d2 = torch.full((H * W,), 1e9)

        # distance from every grid cell to ball
        d_ball = torch.norm(grid_flat - b, dim=-1)            # (G,)

        # space control score:
        #   positive = team1 is closer (controls this cell)
        #   negative = team2 is closer
        #   ball proximity adds bonus to the controlling team
        control = torch.sigmoid(
            1.5 * (d2 - d1) - 0.8 * d_ball
        )   # (G,)

        targets[t] = control.view(H, W)

    return targets   # (T, H, W)


def build_space_targets_from_npz(npz_path: str,
                                  H: int = 24,
                                  W: int = 16) -> torch.Tensor:
    """
    Convenience function: load .npz and return space targets directly.

    Returns: (T, H, W)
    """
    d      = np.load(npz_path)
    team1  = torch.from_numpy(d["players_team1"].astype(np.float32))
    team2  = torch.from_numpy(d["players_team2"].astype(np.float32))
    ball   = torch.from_numpy(d["ball"].astype(np.float32))
    grid   = make_grid(H, W)

    return build_space_targets(team1, team2, ball, grid,
                               already_normalised=False)