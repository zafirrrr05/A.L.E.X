"""
ClipDataset — PyTorch Dataset for the clip_dataset AVI clips.

Folder structure expected:
    data/raw_videos/clip_dataset/
        corner/       *.avi
        foul/         *.avi
        freekick/     *.avi
        goal/         *.avi
        goalkick/     *.avi
        longpass/     *.avi
        ontarget/     *.avi
        penalty/      *.avi
        shortpass/    *.avi
        substitution/ *.avi
        throw-in/     *.avi

Each clip is loaded, resized, and padded/trimmed to CLIP_FRAMES.
Because no player tracker is available, a single "pseudo-player" bounding
box spanning the full frame is created.  When a real tracker is integrated,
replace `_dummy_bboxes()` with tracker output.

Normalisation follows Kinetics-400 stats used for X3D pre-training:
    mean = [0.45, 0.45, 0.45]
    std  = [0.225, 0.225, 0.225]

Dependencies:
    pip install torch torchvision opencv-python
"""

import os
import glob
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ..configs.labels import FOLDER_TO_CLASS, NUM_CLASSES, BACKGROUND
from ..configs.config  import (
    CLIP_FRAMES, IMG_H, IMG_W, MAX_PLAYERS,
    FRAME_RATE, BATCH_SIZE,
)


# Kinetics-400 normalisation constants (used for X3D pre-training)
MEAN = np.array([0.45, 0.45, 0.45], dtype=np.float32)
STD  = np.array([0.225, 0.225, 0.225], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Video loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_video_frames(
    path: str,
    target_frames: int = CLIP_FRAMES,
    target_h: int = IMG_H,
    target_w: int = IMG_W,
) -> Optional[np.ndarray]:
    """
    Load a video clip and return a (T, H, W, 3) float32 numpy array,
    normalised with Kinetics-400 stats.

    Returns None if the video cannot be opened.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (target_w, target_h),
                           interpolation=cv2.INTER_LINEAR)
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        return None

    frames = np.stack(frames, axis=0).astype(np.float32) / 255.0   # T×H×W×3

    # Normalise
    frames = (frames - MEAN) / STD

    # Pad or trim to target_frames
    T = frames.shape[0]
    if T < target_frames:
        pad = np.zeros((target_frames - T, target_h, target_w, 3),
                       dtype=np.float32)
        frames = np.concatenate([frames, pad], axis=0)
    else:
        # Random temporal crop during training
        start = random.randint(0, T - target_frames)
        frames = frames[start: start + target_frames]

    return frames   # target_frames × H × W × 3


# ──────────────────────────────────────────────────────────────────────────────
# Dummy bounding boxes (single full-frame pseudo-player)
# ──────────────────────────────────────────────────────────────────────────────

def _dummy_bboxes(
    T: int,
    max_players: int = MAX_PLAYERS,
) -> np.ndarray:
    """
    Returns a (max_players, T, 4) bbox array.
    Player 0 has a full-frame bbox [0,0,1,1].
    All other players have zero-area boxes (will be masked out).

    REPLACE THIS with real tracker output when available.
    """
    bboxes = np.zeros((max_players, T, 4), dtype=np.float32)
    # Player 0: full frame
    bboxes[0, :, 2] = 1.0   # x2
    bboxes[0, :, 3] = 1.0   # y2
    return bboxes


def _dummy_player_mask(max_players: int = MAX_PLAYERS) -> np.ndarray:
    mask = np.zeros(max_players, dtype=bool)
    mask[0] = True   # only player 0 is valid
    return mask


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class ClipDataset(Dataset):
    """
    PyTorch Dataset for clip-level action classification.

    Each item is a single video clip with:
        clip        : C×T×H×W  float32  (3×50×352×640)
        bboxes      : N×T×4    float32
        labels      : N×T      int64    (class index per frame per player)
        player_mask : N        bool
        class_idx   : int      (clip-level ground truth, for logging)
        video_path  : str
    """

    def __init__(
        self,
        root_dir: str,                    # path to clip_dataset/
        split: str = "train",             # "train" or "val"
        val_fraction: float = 0.15,
        seed: int = 42,
    ):
        self.root = Path(root_dir)
        self.split = split

        all_samples = []
        for folder, cls_idx in FOLDER_TO_CLASS.items():
            folder_path = self.root / folder
            if not folder_path.exists():
                print(f"[ClipDataset] Warning: {folder_path} not found, skipping.")
                continue
            paths = sorted(glob.glob(str(folder_path / "*.avi")))
            for p in paths:
                all_samples.append((p, cls_idx))

        # Deterministic split
        rng = random.Random(seed)
        rng.shuffle(all_samples)
        n_val = max(1, int(len(all_samples) * val_fraction))

        if split == "val":
            self.samples = all_samples[:n_val]
        else:
            self.samples = all_samples[n_val:]

        print(f"[ClipDataset] {split}: {len(self.samples)} clips "
              f"from {len(FOLDER_TO_CLASS)} classes.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[dict]:
        path, cls_idx = self.samples[idx]

        frames = load_video_frames(path)
        if frames is None:
            # Return a blank item — collate_fn will skip it
            return None

        T = CLIP_FRAMES
        # (T, H, W, 3) → (3, T, H, W)
        clip = torch.from_numpy(frames).permute(3, 0, 1, 2)  # float32

        # Bboxes and mask  (single pseudo-player for now)
        bboxes_np  = _dummy_bboxes(T)         # N×T×4
        mask_np    = _dummy_player_mask()     # N  bool

        bboxes      = torch.from_numpy(bboxes_np)              # N×T×4
        player_mask = torch.from_numpy(mask_np)               # N  bool

        # Labels: the clip's class is assigned to every frame of player 0;
        # all other players are set to -1 (ignored in CE loss).
        labels = torch.full((MAX_PLAYERS, T), fill_value=-1, dtype=torch.long)
        labels[0, :] = cls_idx   # player 0 has the ground-truth class

        return {
            "clip":        clip,          # 3×T×H×W
            "bboxes":      bboxes,        # N×T×4
            "labels":      labels,        # N×T
            "player_mask": player_mask,   # N
            "class_idx":   cls_idx,
            "video_path":  path,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Collate — skip None items (broken videos)
# ──────────────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    root_dir: str,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 4,
    val_fraction: float = 0.15,
) -> tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader).
    """
    train_ds = ClipDataset(root_dir, split="train", val_fraction=val_fraction)
    val_ds   = ClipDataset(root_dir, split="val",   val_fraction=val_fraction)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return train_loader, val_loader
