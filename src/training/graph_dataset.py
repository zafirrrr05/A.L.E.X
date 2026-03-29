"""
football_graph_dataset.py  (v2 — matches SequenceBuilder exactly)
──────────────────────────────────────────────────────────────────
Converts .npz clips produced by SequenceBuilder into PyTorch Geometric
Data objects for the Hierarchical Dual GATv2 pipeline.

Confirmed .npz structure (from SequenceBuilder):
    players_team1  : (T, 11, 4)   [cx, cy, vx, vy] — zero-padded rows if <11 players
    players_team2  : (T, 11, 4)   same
    ball           : (T, 4)       [cx, cy, vx, vy]  — zeros if ball not detected
    referee        : (T, 4)       [cx, cy, vx, vy]  — flat vector (NOT (T,1,4))
    start_frame    : scalar       first frame index of the window

Install:
    pip install torch torch_geometric numpy scipy
"""

import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


# ─────────────────────────────────────────────
#  PITCH CONSTANTS
#  Adjust these to match your coordinate system.
#  If cx/cy are already in metres (0–105, 0–68), keep as-is.
#  If they are in pixels, set PITCH_LENGTH/WIDTH to frame dimensions.
# ─────────────────────────────────────────────

PITCH_LENGTH = 105.0
PITCH_WIDTH  =  68.0

NUM_PLAYERS  = 11
T_FRAMES     = 50

TEAM_A_ID = 0
TEAM_B_ID = 1
BALL_ID   = 2


# ─────────────────────────────────────────────
#  ZERO-PAD MASK
#  A player row is padding when ALL four features are exactly 0.
#  We mark it so the graph builder can skip it.
# ─────────────────────────────────────────────

def get_valid_mask(players_raw: np.ndarray) -> np.ndarray:
    """
    players_raw : (11, 4)
    Returns     : (11,) bool — True where the row is a real player
    """
    return ~np.all(players_raw == 0.0, axis=-1)


# ─────────────────────────────────────────────
#  FEATURE ENGINEERING  (10 features per player)
# ─────────────────────────────────────────────

def engineer_player_features(players_raw: np.ndarray,
                              valid_mask:  np.ndarray,
                              ball_xy:     np.ndarray,
                              team_centroid: np.ndarray) -> np.ndarray:
    """
    players_raw   : (11, 4)  [cx, cy, vx, vy]
    valid_mask    : (11,)    True = real player
    ball_xy       : (2,)     [cx, cy]
    team_centroid : (2,)     mean of valid player positions

    Returns       : (11, 10) — padded rows are zeroed out
    """
    cx, cy = players_raw[:, 0], players_raw[:, 1]
    vx, vy = players_raw[:, 2], players_raw[:, 3]

    speed    = np.sqrt(vx**2 + vy**2)
    angle    = np.arctan2(vy, vx)
    d_ball_x = cx - ball_xy[0]
    d_ball_y = cy - ball_xy[1]
    d_ball   = np.sqrt(d_ball_x**2 + d_ball_y**2)
    d_cent   = np.sqrt((cx - team_centroid[0])**2 +
                       (cy - team_centroid[1])**2)

    cx_n = cx / PITCH_LENGTH
    cy_n = cy / PITCH_WIDTH

    feats = np.stack([
        cx_n,                        # 0  normalised x
        cy_n,                        # 1  normalised y
        vx,                          # 2  velocity x
        vy,                          # 3  velocity y
        speed,                       # 4  speed magnitude
        angle,                       # 5  direction of motion (radians)
        d_ball_x / PITCH_LENGTH,     # 6  relative x to ball
        d_ball_y / PITCH_WIDTH,      # 7  relative y to ball
        d_ball   / PITCH_LENGTH,     # 8  distance to ball
        d_cent   / PITCH_LENGTH,     # 9  distance to team centroid
    ], axis=-1).astype(np.float32)   # (11, 10)

    # zero out padded rows so they don't inject noise
    feats[~valid_mask] = 0.0
    return feats


def engineer_ball_features(ball_raw: np.ndarray) -> np.ndarray:
    """
    ball_raw : (4,)  [cx, cy, vx, vy]
    Returns  : (10,) — zero-padded to match player feature width
    """
    cx, cy, vx, vy = ball_raw
    speed = math.sqrt(float(vx)**2 + float(vy)**2)
    fb = np.array([
        cx / PITCH_LENGTH,
        cy / PITCH_WIDTH,
        vx, vy, speed,
        0.0, 0.0, 0.0, 0.0, 0.0   # unused slots
    ], dtype=np.float32)
    return fb


# ─────────────────────────────────────────────
#  EDGE BUILDERS
# ─────────────────────────────────────────────

def knn_edges(xy: np.ndarray, valid_idx: np.ndarray, k: int = 5):
    """
    Build undirected KNN edges among valid nodes only.

    xy        : (N, 2) — positions of ALL slots (including padding)
    valid_idx : (V,)   — indices of valid (non-padded) nodes
    k         : neighbours per node

    Returns   : (2, E) int64
    """
    if len(valid_idx) < 2:
        return np.zeros((2, 0), dtype=np.int64)

    xy_v = xy[valid_idx]
    V    = len(valid_idx)
    k    = min(k, V - 1)

    dists = np.sum((xy_v[:, None] - xy_v[None, :]) ** 2, axis=-1)
    np.fill_diagonal(dists, np.inf)

    src, dst = [], []
    for i in range(V):
        for j in np.argsort(dists[i])[:k]:
            src += [int(valid_idx[i]), int(valid_idx[j])]
            dst += [int(valid_idx[j]), int(valid_idx[i])]

    edges = np.array([src, dst], dtype=np.int64)
    return np.unique(edges, axis=1)


def cross_team_edges(xy1: np.ndarray, valid1: np.ndarray,
                     xy2: np.ndarray, valid2: np.ndarray,
                     k: int = 3, offset: int = 11):
    """
    Connect each valid player in team A to their k nearest opponents in team B.
    offset : node index offset for team B (always 11 in our layout)
    Returns: (2, E) int64
    """
    if len(valid1) == 0 or len(valid2) == 0:
        return np.zeros((2, 0), dtype=np.int64)

    k = min(k, len(valid2))
    src, dst = [], []

    for i in valid1:
        dists_to_b = np.sum((xy1[i] - xy2[valid2]) ** 2, axis=-1)
        nearest = valid2[np.argsort(dists_to_b)[:k]]
        for j in nearest:
            src += [int(i),          int(j) + offset]
            dst += [int(j) + offset, int(i)         ]

    edges = np.array([src, dst], dtype=np.int64)
    return np.unique(edges, axis=1)


def ball_edges(xy_all: np.ndarray, valid_players: np.ndarray,
               ball_node_idx: int, k: int = 6):
    """
    Connect ball node to k nearest valid players (undirected).
    """
    if len(valid_players) == 0:
        return np.zeros((2, 0), dtype=np.int64)

    k = min(k, len(valid_players))
    ball_pos = xy_all[ball_node_idx]
    dists    = np.sum((xy_all[valid_players] - ball_pos) ** 2, axis=-1)
    nearest  = valid_players[np.argsort(dists)[:k]]

    src = np.concatenate([nearest,               np.full(k, ball_node_idx)])
    dst = np.concatenate([np.full(k, ball_node_idx), nearest              ])
    return np.unique(np.array([src, dst], dtype=np.int64), axis=1)


def build_edge_features(xy: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    """
    4 edge features: [norm_dist, delta_x, delta_y, same_third]
    """
    if edge_index.shape[1] == 0:
        return np.zeros((0, 4), dtype=np.float32)

    s, d  = edge_index[0], edge_index[1]
    dx    = xy[d, 0] - xy[s, 0]
    dy    = xy[d, 1] - xy[s, 1]
    dist  = np.sqrt(dx**2 + dy**2)

    zone_s     = (xy[s, 0] / PITCH_LENGTH * 3).astype(int).clip(0, 2)
    zone_d     = (xy[d, 0] / PITCH_LENGTH * 3).astype(int).clip(0, 2)
    same_third = (zone_s == zone_d).astype(np.float32)

    return np.stack([
        dist / PITCH_LENGTH,
        dx   / PITCH_LENGTH,
        dy   / PITCH_WIDTH,
        same_third,
    ], axis=-1).astype(np.float32)


# ─────────────────────────────────────────────
#  SINGLE FRAME → PyG Data
# ─────────────────────────────────────────────

def frame_to_player_graph(t1_raw: np.ndarray,
                           t2_raw: np.ndarray,
                           ball_raw: np.ndarray) -> Data:
    """
    Build the 23-node player-level graph for one frame.

    Node layout:
        0  – 10  : Team A players  (some may be zero-padded)
        11 – 21  : Team B players  (some may be zero-padded)
        22       : Ball
    """
    ball_xy = ball_raw[:2]

    # valid player masks
    mask1  = get_valid_mask(t1_raw)
    mask2  = get_valid_mask(t2_raw)
    valid1 = np.where(mask1)[0]      # indices 0–10
    valid2 = np.where(mask2)[0]      # indices 0–10

    # team centroids over real players only
    centroid1 = t1_raw[valid1, :2].mean(0) if len(valid1) > 0 else ball_xy
    centroid2 = t2_raw[valid2, :2].mean(0) if len(valid2) > 0 else ball_xy

    # node features
    f1 = engineer_player_features(t1_raw, mask1, ball_xy, centroid1)  # (11,10)
    f2 = engineer_player_features(t2_raw, mask2, ball_xy, centroid2)  # (11,10)
    fb = engineer_ball_features(ball_raw)                              # (10,)

    node_feats = np.vstack([f1, f2, fb[None]])   # (23, 10)

    # team identity
    team_ids = np.array(
        [TEAM_A_ID]*11 + [TEAM_B_ID]*11 + [BALL_ID],
        dtype=np.int64
    )

    # global indices
    valid1_global      = valid1
    valid2_global      = valid2 + 11
    all_valid_players  = np.concatenate([valid1_global, valid2_global])
    ball_node          = 22

    # edges
    e_intra1 = knn_edges(t1_raw[:, :2], valid1, k=5)
    e_intra2 = knn_edges(t2_raw[:, :2], valid2, k=5) + 11
    e_cross  = cross_team_edges(t1_raw[:, :2], valid1,
                                t2_raw[:, :2], valid2,
                                k=3, offset=11)
    all_xy   = np.vstack([t1_raw[:, :2], t2_raw[:, :2], ball_xy[None]])
    e_ball   = ball_edges(all_xy, all_valid_players, ball_node, k=6)

    edge_index = np.concatenate([e_intra1, e_intra2, e_cross, e_ball], axis=1)
    edge_index = np.unique(edge_index, axis=1)
    edge_attr  = build_edge_features(all_xy, edge_index)

    # valid node mask for loss masking downstream
    valid_nodes = np.zeros(23, dtype=bool)
    valid_nodes[all_valid_players] = True
    valid_nodes[ball_node]         = True

    return Data(
        x           = torch.from_numpy(node_feats),
        edge_index  = torch.from_numpy(edge_index),
        edge_attr   = torch.from_numpy(edge_attr),
        team_ids    = torch.from_numpy(team_ids),
        valid_nodes = torch.from_numpy(valid_nodes),
    )


def frame_to_team_graph(t1_raw: np.ndarray,
                         t2_raw: np.ndarray,
                         ball_raw: np.ndarray) -> Data:
    """
    Build the 3-node team-level graph for one frame.
    Nodes: [Team A, Team B, Ball]
    7 features per node (hand-crafted statistics).
    G1 embeddings are injected later in the model forward pass.
    """
    def team_stats(raw):
        valid = get_valid_mask(raw)
        xy    = raw[valid, :2]
        v     = raw[valid, 2:]
        if len(xy) == 0:
            return np.zeros(7, dtype=np.float32)
        centroid  = xy.mean(0)
        spread    = xy.std(0) if len(xy) > 1 else np.zeros(2)
        avg_speed = np.linalg.norm(v, axis=1).mean()
        d_ball    = np.linalg.norm(xy - ball_raw[:2], axis=1).min()
        n_valid   = len(xy) / 11.0
        return np.array([
            centroid[0] / PITCH_LENGTH,
            centroid[1] / PITCH_WIDTH,
            spread[0]   / PITCH_LENGTH,
            spread[1]   / PITCH_WIDTH,
            avg_speed,
            d_ball      / PITCH_LENGTH,
            n_valid,
        ], dtype=np.float32)

    bx    = float(ball_raw[0]) / PITCH_LENGTH
    by    = float(ball_raw[1]) / PITCH_WIDTH
    bvx   = float(ball_raw[2])
    bvy   = float(ball_raw[3])
    bspd  = math.sqrt(bvx**2 + bvy**2)
    n_ball = np.array([bx, by, 0.0, 0.0, bspd, 0.0, 1.0], dtype=np.float32)

    x = torch.from_numpy(
        np.stack([team_stats(t1_raw), team_stats(t2_raw), n_ball])
    )  # (3, 7)

    edge_index = torch.tensor([
        [0, 0, 1, 1, 2, 2],
        [1, 2, 0, 2, 0, 1]
    ], dtype=torch.long)

    return Data(x=x, edge_index=edge_index)


# ─────────────────────────────────────────────
#  CLIP → SEQUENCE OF FRAME GRAPHS
# ─────────────────────────────────────────────

def clip_to_graph_sequence(npz_path: str, label: int = None):
    """
    Load one .npz clip and return:
        sequence    : list of (player_graph, team_graph) — one per frame
        start_frame : int
    """
    data = np.load(npz_path)
    t1   = data['players_team1']   # (T, 11, 4)
    t2   = data['players_team2']   # (T, 11, 4)
    ball = data['ball']            # (T, 4)
    # referee = data['referee']    # (T, 4) — available if you want to add it
    T    = t1.shape[0]

    sequence = []
    for t in range(T):
        pg = frame_to_player_graph(t1[t], t2[t], ball[t])
        tg = frame_to_team_graph(t1[t], t2[t], ball[t])
        if label is not None:
            pg.y = torch.tensor([label], dtype=torch.long)
            tg.y = torch.tensor([label], dtype=torch.long)
        sequence.append((pg, tg))

    return sequence, int(data['start_frame'])


# ─────────────────────────────────────────────
#  TRAIN / VAL SPLIT  (respects sliding window overlap)
# ─────────────────────────────────────────────

def split_clips_by_start_frame(npz_paths: list,
                                val_ratio: float = 0.2,
                                gap: int = 50):
    """
    Splits clips so that val clips are at least `gap` frames away
    from the nearest train clip — avoids leakage from the 40-frame overlap.
    Clips are sorted by start_frame before splitting.
    """
    clips = []
    for p in npz_paths:
        d = np.load(p)
        clips.append((int(d['start_frame']), p))
    clips.sort(key=lambda x: x[0])

    n_val         = max(1, int(len(clips) * val_ratio))
    val_start_idx = len(clips) - n_val

    # walk back until we have at least `gap` frames between train and val
    while val_start_idx > 0:
        if clips[val_start_idx][0] - clips[val_start_idx - 1][0] >= gap:
            break
        val_start_idx -= 1

    train_paths = [p for _, p in clips[:val_start_idx]]
    val_paths   = [p for _, p in clips[val_start_idx:]]
    return train_paths, val_paths


# ─────────────────────────────────────────────
#  PYTORCH DATASET
# ─────────────────────────────────────────────

class FootballTrackingDataset(Dataset):
    """
    Wraps a list of .npz clip paths.
    labels : dict {filename_stem: int_label}  or  None
    """

    def __init__(self, npz_paths: list, labels: dict = None):
        self.paths  = npz_paths
        self.labels = labels or {}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path  = self.paths[idx]
        stem  = os.path.splitext(os.path.basename(path))[0]
        label = self.labels.get(stem, None)
        seq, start_frame = clip_to_graph_sequence(path, label)
        return seq, start_frame


# ─────────────────────────────────────────────
#  QUICK SANITY CHECK
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python football_graph_dataset.py path/to/clip.npz")
        sys.exit(1)

    seq, sf = clip_to_graph_sequence(sys.argv[1])
    pg, tg  = seq[0]

    print(f"Clip loaded : {len(seq)} frames  |  start_frame={sf}")
    print(f"\n[Frame 0] Player graph")
    print(f"  x          : {tuple(pg.x.shape)}   — 23 nodes, 10 features")
    print(f"  edge_index : {tuple(pg.edge_index.shape)}")
    print(f"  edge_attr  : {tuple(pg.edge_attr.shape)}")
    print(f"  valid nodes: {pg.valid_nodes.sum().item()} / 23")

    print(f"\n[Frame 0] Team graph")
    print(f"  x          : {tuple(tg.x.shape)}   — 3 nodes, 7 features")
    print(f"  edge_index : {tuple(tg.edge_index.shape)}")

    assert not torch.isnan(pg.x).any(), "NaN in player node features!"
    assert not torch.isnan(tg.x).any(), "NaN in team node features!"
    print("\nAll checks passed.")
