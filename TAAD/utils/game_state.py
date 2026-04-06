"""
Game State Builder — proxy implementation.

Until homography + 2-D pitch reconstruction is available, this module
derives player positions and velocities from screen-space bounding box
centres (normalised to [0, 1]).

HOW TO UPGRADE:
    When homography is ready, replace the body of `extract_game_state()`
    with real pitch coordinate extraction.  The output contract (dict of
    tensors) must stay identical so that TAADWithGNN.forward() requires
    zero changes.

Proxy convention
----------------
  position  = normalised centre of bbox  (cx/W, cy/H)  ∈ [0, 1]²
  velocity  = finite-difference of positions over adjacent frames, zero-padded
  team_id   = 0 for all players (no tracking data yet; override once available)
  player_mask = True for every valid bbox (False for padding rows)
"""

import torch
import numpy as np
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Core extraction
# ──────────────────────────────────────────────────────────────────────────────

def bboxes_to_centres(bboxes: torch.Tensor) -> torch.Tensor:
    """
    Convert [x1, y1, x2, y2] normalised bboxes to centre coordinates.

    Args:
        bboxes: B×N×T×4  (normalised [0,1])
    Returns:
        centres: B×N×T×2  (cx, cy)
    """
    cx = (bboxes[..., 0] + bboxes[..., 2]) / 2.0
    cy = (bboxes[..., 1] + bboxes[..., 3]) / 2.0
    return torch.stack([cx, cy], dim=-1)   # B×N×T×2


def centres_to_velocity(centres: torch.Tensor) -> torch.Tensor:
    """
    Finite-difference velocity from consecutive frame positions.
    Frame 0 velocity is set to 0.

    Args:
        centres: B×N×T×2
    Returns:
        velocity: B×N×T×2
    """
    diff = centres[..., 1:, :] - centres[..., :-1, :]  # B×N×(T-1)×2
    # Pad first frame with zero velocity
    pad  = torch.zeros_like(centres[..., :1, :])        # B×N×1×2
    return torch.cat([pad, diff], dim=2)                 # B×N×T×2


def extract_game_state(
    bboxes: torch.Tensor,                # B×N×T×4  normalised
    team_ids: Optional[torch.Tensor] = None,   # B×N  (0 or 1) — if known
    player_mask: Optional[torch.Tensor] = None, # B×N  bool
    use_proxy: bool = True,
) -> dict:
    """
    Build the game-state tensors required by LocalGameStateGNN.

    Args:
        bboxes      : B×N×T×4   normalised [0,1] bounding boxes
        team_ids    : B×N        team membership (0/1).  If None, zeros used.
        player_mask : B×N        True for valid players.  If None, all True.
        use_proxy   : if True, derive positions from bboxes (proxy mode).
                      Set False when real pitch coords are provided externally.

    Returns:
        dict with keys:
            "positions"   → B×N×T×2   (float32)
            "velocities"  → B×N×T×2   (float32)
            "team_ids"    → B×N×T     (int64)
            "player_mask" → B×N       (bool)
    """
    B, N, T, _ = bboxes.shape
    device = bboxes.device

    if use_proxy:
        positions  = bboxes_to_centres(bboxes)     # B×N×T×2
        velocities = centres_to_velocity(positions) # B×N×T×2
    else:
        # Real pitch coords should be passed in as `bboxes` parameter
        # when use_proxy=False (caller responsibility).
        positions  = bboxes[..., :2]
        velocities = bboxes[..., 2:]

    if team_ids is None:
        # All zeros — no tracking data yet
        team_ids_t = torch.zeros(B, N, T, dtype=torch.int64, device=device)
    else:
        # Expand B×N → B×N×T
        team_ids_t = team_ids.unsqueeze(-1).expand(B, N, T).long()

    if player_mask is None:
        player_mask = torch.ones(B, N, dtype=torch.bool, device=device)

    return {
        "positions":    positions,    # B×N×T×2
        "velocities":   velocities,   # B×N×T×2
        "team_ids":     team_ids_t,   # B×N×T
        "player_mask":  player_mask,  # B×N
    }


# ──────────────────────────────────────────────────────────────────────────────
# Dummy bbox generator  (for testing / when no tracker is available)
# ──────────────────────────────────────────────────────────────────────────────

def dummy_bboxes(
    B: int, N: int, T: int,
    img_h: int, img_w: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generates random normalised bboxes for unit-testing.
    Replace with real tracker output in production.

    Returns:
        bboxes: B×N×T×4  (x1,y1,x2,y2) ∈ [0,1]
    """
    # Random top-left corner and size
    x1 = torch.rand(B, N, T, device=device) * 0.8
    y1 = torch.rand(B, N, T, device=device) * 0.8
    w  = torch.rand(B, N, T, device=device) * 0.15 + 0.02
    h  = torch.rand(B, N, T, device=device) * 0.15 + 0.04
    x2 = (x1 + w).clamp(0, 1)
    y2 = (y1 + h).clamp(0, 1)
    return torch.stack([x1, y1, x2, y2], dim=-1)
