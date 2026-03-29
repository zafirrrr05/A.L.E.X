"""
scripts/train_space_control.py  (v3 — with quality filter)
────────────────────────────────────────────────────────────
Trains the space control head of HierarchicalDualGATv2.

Changes vs v2:
  - is_valid_sequence() filter added to SpaceControlDataset
  - _get_clip_embedding() simplified and fixed
  - SpaceHead added as a standalone module above GATv2 heads
  - max_clips applies AFTER filtering

Saves: checkpoints/space_gatv2.pt
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from src.models.dual_gatv2_model import HierarchicalDualGATv2
from src.training.graph_dataset import frame_to_player_graph, frame_to_team_graph
from src.training.space_targets import build_space_targets, make_grid
from src.utils.data_utils import is_valid_sequence

PITCH_W  = 1280.0
PITCH_H  = 768.0
VEL_CLIP = 5.0
GRID_H   = 24
GRID_W   = 16


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


# ─────────────────────────────────────────────
#  SPACE HEAD
# ─────────────────────────────────────────────

class SpaceHead(nn.Module):
    """
    Predicts a (H, W) space control map from the 256-dim clip embedding.
    """
    def __init__(self, in_dim: int = 256, H: int = 24, W: int = 16):
        super().__init__()
        self.H = H
        self.W = W
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, H * W),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.mlp(z).view(-1, self.H, self.W)


# ─────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────

class SpaceControlDataset(Dataset):
    """
    Each sample: full 50-frame clip → mean space control map (H, W).
    Filters warmup/low-quality sequences automatically.
    """

    def __init__(self, root_dir: str, max_clips: int = None):
        self.samples = []
        grid = make_grid(GRID_H, GRID_W)

        files = sorted([
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith(".npz")
        ])

        n_filtered = 0
        n_loaded   = 0

        for fpath in tqdm(files, desc="Building space dataset"):

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

            # space targets (T, H, W) → mean over time
            space  = build_space_targets(t1_t, t2_t, ball_t, grid,
                                         already_normalised=True)
            target = space.mean(dim=0)   # (H, W)

            # build graph sequence
            T        = t1.shape[0]
            sequence = []
            for f in range(T):
                pg = frame_to_player_graph(t1[f], t2[f], ball[f])
                tg = frame_to_team_graph(t1[f],  t2[f], ball[f])
                sequence.append((pg, tg))

            self.samples.append({"sequence": sequence, "target": target})
            n_loaded += 1

            if max_clips is not None and n_loaded >= max_clips:
                break

        print(f"SpaceControlDataset: {n_loaded} clips  "
              f"(filtered={n_filtered})  grid={GRID_H}x{GRID_W}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


# ─────────────────────────────────────────────
#  COLLATE
# ─────────────────────────────────────────────

def space_collate(batch):
    sequences = [item["sequence"] for item in batch]
    targets   = torch.stack([item["target"] for item in batch])
    return sequences, targets


# ─────────────────────────────────────────────
#  EMBEDDING EXTRACTION
# ─────────────────────────────────────────────

def get_clip_embedding(backbone: HierarchicalDualGATv2,
                       clips: list,
                       device: str) -> torch.Tensor:
    """Extract 256-dim clip embedding without running task heads."""
    frame_embs = []
    for clip in clips:
        clip_frames = []
        for pg, tg in clip:
            fe = backbone._process_frame(pg, tg)   # (1, frame_dim)
            clip_frames.append(fe)
        seq = torch.cat(clip_frames, dim=0)         # (T, frame_dim)
        frame_embs.append(seq)

    x_seq = torch.stack(frame_embs, dim=0)          # (B, T, frame_dim)
    z     = backbone.temporal(x_seq)                # (B, 256)
    return z


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Space Control] device = {device}")

    # ── dataset — start with 5000 clips for speed ──
    ds = SpaceControlDataset("data/sequences", max_clips=5000)

    n_val   = max(1, int(len(ds) * 0.2))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              collate_fn=space_collate, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              collate_fn=space_collate, num_workers=0)

    # ── backbone (frozen) ──
    backbone = HierarchicalDualGATv2(
        player_in=10, player_edge_dim=4, player_hidden=64,
        player_heads=4, team_hidden=64, lstm_hidden=128,
        formation_classes=5, set_piece_classes=4,
    ).to(device)

    ckpt_path = "checkpoints/pass_gatv2.pt"
    if os.path.exists(ckpt_path):
        backbone.load_state_dict(
            torch.load(ckpt_path, map_location=device), strict=False
        )
        print(f"Loaded backbone from {ckpt_path}")
    else:
        print("No checkpoint found — training space head from scratch")

    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()

    # ── space head (trainable) ──
    space_head = SpaceHead(in_dim=256, H=GRID_H, W=GRID_W).to(device)
    print(f"Space head params: {sum(p.numel() for p in space_head.parameters()):,}")

    optim     = torch.optim.Adam(space_head.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.5)
    mse       = nn.MSELoss()

    os.makedirs("checkpoints", exist_ok=True)
    best_val = float("inf")

    for epoch in range(15):

        # ── train ──
        space_head.train()
        train_loss = 0.0

        for sequences, targets in tqdm(
            train_loader, desc=f"Epoch {epoch+1:02d} train", leave=False
        ):
            targets = targets.to(device)

            clips = [[(pg.to(device), tg.to(device)) for pg, tg in seq]
                     for seq in sequences]

            with torch.no_grad():
                z = get_clip_embedding(backbone, clips, device)

            pred = space_head(z)
            loss = mse(pred, targets)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(space_head.parameters(), 1.0)
            optim.step()

            train_loss += loss.item()

        scheduler.step()

        # ── val ──
        space_head.eval()
        val_loss = 0.0

        with torch.no_grad():
            for sequences, targets in val_loader:
                targets = targets.to(device)
                clips   = [[(pg.to(device), tg.to(device)) for pg, tg in seq]
                           for seq in sequences]
                z    = get_clip_embedding(backbone, clips, device)
                pred = space_head(z)
                val_loss += mse(pred, targets).item()

        avg_train = train_loss / max(len(train_loader), 1)
        avg_val   = val_loss   / max(len(val_loader),   1)

        print(f"Epoch {epoch+1:02d} | "
              f"train MSE {avg_train:.5f} | val MSE {avg_val:.5f}")

        if avg_val < best_val:
            best_val = avg_val
            torch.save({
                "space_head": space_head.state_dict(),
                "backbone":   backbone.state_dict(),
            }, "checkpoints/space_gatv2.pt")
            print(f"  → saved (val MSE {avg_val:.5f})")

    print("Done. Best checkpoint: checkpoints/space_gatv2.pt")


if __name__ == "__main__":
    main()