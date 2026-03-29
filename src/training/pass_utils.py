"""
src/training/pass_utils.py  (v3)
────────────────────────────────
Pass detection and geometric quality labelling.

Key change vs v2:
  - pass_success() removed — was too noisy (future-frame possession tracking)
  - pass_quality_label() added — labels based on spatial geometry AT pass moment:
      pressure on passer + space available to receivers
  - compute_pass_pressure() kept as standalone utility
  - All functions use normalised [0,1] coords (after ssl_dataset normalisation)
"""

import torch


# ─────────────────────────────────────────────
#  POSSESSION UTILITY
# ─────────────────────────────────────────────

def possessor(team1_t: torch.Tensor,
              team2_t: torch.Tensor,
              ball_t:  torch.Tensor):
    """
    Find which player (across both teams) is closest to the ball.

    team1_t : (11, 4)
    team2_t : (11, 4)
    ball_t  : (4,)

    Returns : (team_id, player_idx)
              team_id    = 0 (team1) or 1 (team2)
              player_idx = 0-10 within that team
    """
    ball_xy = ball_t[:2]

    valid1 = ~torch.all(team1_t == 0, dim=-1)
    valid2 = ~torch.all(team2_t == 0, dim=-1)

    d1 = torch.norm(team1_t[:, :2] - ball_xy, dim=1)
    d2 = torch.norm(team2_t[:, :2] - ball_xy, dim=1)

    d1[~valid1] = 1e9
    d2[~valid2] = 1e9

    min1, idx1 = d1.min(0)
    min2, idx2 = d2.min(0)

    if min1 <= min2:
        return 0, idx1.item()
    else:
        return 1, idx2.item()


def possessor_team(team1_t: torch.Tensor,
                   team2_t: torch.Tensor,
                   ball_t:  torch.Tensor) -> int:
    """Convenience — returns just team_id (0 or 1)."""
    team_id, _ = possessor(team1_t, team2_t, ball_t)
    return team_id


# ─────────────────────────────────────────────
#  PASS EVENT DETECTION
# ─────────────────────────────────────────────

def detect_pass_events(team1: torch.Tensor,
                       team2: torch.Tensor,
                       ball:  torch.Tensor,
                       min_ball_speed: float = 0.02,
                       min_gap:        int   = 5):
    """
    Detect frames where a pass likely occurred.

    team1 : (T, 11, 4)
    team2 : (T, 11, 4)
    ball  : (T, 4)      — must be normalised before calling

    Returns: list of frame indices
    """
    events = []
    last_event_t = -999
    prev_team, _ = possessor(team1[0], team2[0], ball[0])

    for t in range(1, team1.shape[0]):
        curr_team, _ = possessor(team1[t], team2[t], ball[t])
        speed = torch.norm(ball[t, 2:4]).item()

        if curr_team != prev_team and speed > min_ball_speed:
            if t - last_event_t >= min_gap:
                events.append(t)
                last_event_t = t

        prev_team = curr_team

    return events


# ─────────────────────────────────────────────
#  PASS WINDOW EXTRACTION
# ─────────────────────────────────────────────

def extract_pass_window(team1: torch.Tensor,
                        team2: torch.Tensor,
                        ball:  torch.Tensor,
                        t:     int,
                        pre:   int = 5,
                        post:  int = 10):
    """
    Extract a (pre + post) frame window centred on pass event t.
    Returns None if the window falls outside the sequence bounds.
    """
    start = t - pre
    end   = t + post

    if start < 0 or end >= team1.shape[0]:
        return None

    return {
        "team1": team1[start:end],   # (pre+post, 11, 4)
        "team2": team2[start:end],
        "ball":  ball[start:end],
        "t_rel": pre,
    }


# ─────────────────────────────────────────────
#  GEOMETRIC PASS QUALITY LABEL  (replaces pass_success)
# ─────────────────────────────────────────────

def pass_quality_label(team1:  torch.Tensor,
                       team2:  torch.Tensor,
                       ball:   torch.Tensor,
                       t_pass: int,
                       pressure_radius:   float = 0.08,
                       min_receiver_space: float = 0.15) -> int | None:
    """
    Label a pass as good (1) or bad (0) based on spatial geometry
    AT the moment of the pass. No future frames needed.

    A pass is GOOD when:
      - Passer is under LOW defensive pressure   (few defenders nearby)
      - Receivers have OPEN space               (far from nearest defender)

    A pass is BAD when:
      - Passer is under HIGH pressure            (defenders crowding)
      - Receivers are CLOSED down               (defenders tight on receivers)

    team1, team2 : (T, 11, 4)  — must be normalised [0,1] before calling
    ball         : (T, 4)
    t_pass       : frame index of the pass event
    pressure_radius    : normalised distance threshold for "pressured"
                         0.12 ≈ ~154px / ~12m on a real pitch
    min_receiver_space : normalised distance threshold for "open"
                         0.08 ≈ ~102px / ~8m on a real pitch

    Returns: 1 (good pass), 0 (bad pass), or None if not computable
    """
    team_id, passer_idx = possessor(team1[t_pass], team2[t_pass], ball[t_pass])

    if team_id == 0:
        passer_pos = team1[t_pass, passer_idx, :2]   # (2,)
        defenders  = team2[t_pass]                    # (11, 4) — opponents
        receivers  = team1[t_pass]                    # (11, 4) — teammates
    else:
        passer_pos = team2[t_pass, passer_idx, :2]
        defenders  = team1[t_pass]
        receivers  = team2[t_pass]

    # ── 1. pressure on the passer ──
    valid_def = ~torch.all(defenders == 0, dim=-1)    # (11,) bool
    if valid_def.sum() == 0:
        return None

    d_to_passer = torch.norm(
        defenders[valid_def, :2] - passer_pos, dim=1
    )   # (V_def,)
    pressure = (d_to_passer < pressure_radius).float().mean().item()

    # ── 2. space available to receivers ──
    # exclude the passer themselves from the receiver list
    valid_rec = ~torch.all(receivers == 0, dim=-1)
    valid_rec[passer_idx] = False                     # passer can't receive

    if valid_rec.sum() == 0:
        return None

    rec_pos  = receivers[valid_rec, :2]               # (V_rec, 2)
    def_pos  = defenders[valid_def, :2]               # (V_def, 2)

    # for each receiver, find distance to their nearest defender
    diffs        = rec_pos[:, None, :] - def_pos[None, :, :]   # (R, D, 2)
    min_def_dist = torch.norm(diffs, dim=-1).min(dim=1)[0]      # (R,)
    avg_space    = min_def_dist.mean().item()

    # ── 3. label ──
    # good = passer under low pressure AND receivers have space
    is_good = int(pressure < 0.15 and avg_space > 0.15)

    return is_good


# ─────────────────────────────────────────────
#  STANDALONE PRESSURE UTILITY
# ─────────────────────────────────────────────

def compute_pass_pressure(team1_t: torch.Tensor,
                          team2_t: torch.Tensor,
                          ball_t:  torch.Tensor,
                          pressure_radius: float = 0.12) -> float:
    """
    Compute defensive pressure on the ball carrier at a single frame.
    Returns float in [0, 1].
    """
    team_id, passer_idx = possessor(team1_t, team2_t, ball_t)

    if team_id == 0:
        passer_pos = team1_t[passer_idx, :2]
        defenders  = team2_t
    else:
        passer_pos = team2_t[passer_idx, :2]
        defenders  = team1_t

    valid = ~torch.all(defenders == 0, dim=-1)
    if valid.sum() == 0:
        return 0.0

    dists    = torch.norm(defenders[valid, :2] - passer_pos, dim=1)
    pressure = (dists < pressure_radius).float().mean().item()
    return pressure