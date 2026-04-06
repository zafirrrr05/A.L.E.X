"""
Training script — TAAD + GNN Action Detection Module.

Implements the training procedure from Section 3.6:
  - Adam optimiser (lr=5e-4, weight_decay=1e-5 on non-bias params)
  - Gradient accumulation over 20 iterations
  - Batch size 6
  - 13 epochs total, LR ÷ 10 at epoch 10
  - Cross-entropy loss

Usage:
    python -m TAAD.train \
        --data  data/raw_videos/clip_dataset \
        --ckpt  checkpoints/taad_gnn.pt \
        [--device cuda]

Dependencies:
    pip install torch torchvision pytorchvideo torch-geometric tqdm
"""

import argparse
import os
import time
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

# ── local imports ─────────────────────────────────────────────────────────────
from .configs.config  import (
    NUM_EPOCHS, LR, WEIGHT_DECAY, GRAD_ACCUM_STEPS,
    LR_DROP_EPOCH, BATCH_SIZE, USE_PROXY_POSITIONS,
)
from .configs.labels  import NUM_CLASSES
from .models.taad_gnn import TAADWithGNN
from .utils.dataset   import build_dataloaders
from .utils.game_state import extract_game_state
from .configs import config as cfg_module


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation (mAP stub — temporal IoU mAP requires the full tube matcher)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(model, loader, device):
    """
    Computes validation cross-entropy loss.
    Full mAP (Section 4.1) requires the tube-matching pipeline; add that
    once the inference runner is integrated.
    """
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue

            clip        = batch["clip"].to(device)         # B×3×T×H×W
            bboxes      = batch["bboxes"].to(device)       # B×N×T×4
            labels      = batch["labels"].to(device)       # B×N×T
            player_mask = batch["player_mask"].to(device)  # B×N

            gs = extract_game_state(
                bboxes, player_mask=player_mask,
                use_proxy=USE_PROXY_POSITIONS,
            )

            out = model(
                clip        = clip,
                bboxes      = bboxes,
                positions   = gs["positions"].to(device),
                velocities  = gs["velocities"].to(device),
                team_ids    = gs["team_ids"].to(device),
                player_mask = gs["player_mask"].to(device),
                labels      = labels,
            )
            total_loss += out["loss"].item()
            n_batches  += 1

    return total_loss / max(n_batches, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    # Model
    model = TAADWithGNN(cfg=cfg_module, num_classes=NUM_CLASSES).to(device)

    # Optimiser with separate weight-decay groups (Section 3.6)
    param_groups = model.get_param_groups(weight_decay=WEIGHT_DECAY)
    optimiser    = Adam(param_groups, lr=LR)

    # LR schedule: divide by 10 at epoch LR_DROP_EPOCH
    scheduler = MultiStepLR(optimiser, milestones=[LR_DROP_EPOCH], gamma=0.1)

    # Dataloaders
    train_loader, val_loader = build_dataloaders(
        root_dir=args.data,
        batch_size=BATCH_SIZE,
        num_workers=args.workers,
    )

    ckpt_dir = Path(args.ckpt).parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss  = 0.0
        n_batches   = 0
        accum_count = 0

        optimiser.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")

        for batch in pbar:
            if batch is None:
                continue

            clip        = batch["clip"].to(device)
            bboxes      = batch["bboxes"].to(device)
            labels      = batch["labels"].to(device)
            player_mask = batch["player_mask"].to(device)

            # Build game-state tensors (proxy or real)
            gs = extract_game_state(
                bboxes, player_mask=player_mask,
                use_proxy=USE_PROXY_POSITIONS,
            )

            out = model(
                clip        = clip,
                bboxes      = bboxes,
                positions   = gs["positions"].to(device),
                velocities  = gs["velocities"].to(device),
                team_ids    = gs["team_ids"].to(device),
                player_mask = gs["player_mask"].to(device),
                labels      = labels,
            )

            loss = out["loss"] / GRAD_ACCUM_STEPS
            loss.backward()
            accum_count += 1

            epoch_loss += out["loss"].item()
            n_batches  += 1

            if accum_count == GRAD_ACCUM_STEPS:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimiser.step()
                optimiser.zero_grad()
                accum_count = 0

            pbar.set_postfix({"loss": f"{out['loss'].item():.4f}"})

        # Final gradient update for remaining accumulation steps
        if accum_count > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimiser.step()
            optimiser.zero_grad()

        scheduler.step()

        avg_train = epoch_loss / max(n_batches, 1)
        avg_val   = evaluate(model, val_loader, device)

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={avg_train:.4f}  "
            f"val_loss={avg_val:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        # Checkpoint
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "optimiser":  optimiser.state_dict(),
                "scheduler":  scheduler.state_dict(),
                "val_loss":   avg_val,
            }, args.ckpt)
            print(f"  ✓ Saved best checkpoint → {args.ckpt}")

    print(f"[Train] Done. Best val loss: {best_val_loss:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train TAAD+GNN action detector")
    p.add_argument("--data",    required=True,
                   help="Path to clip_dataset root folder")
    p.add_argument("--ckpt",    default="checkpoints/taad_gnn.pt",
                   help="Path to save best checkpoint")
    p.add_argument("--device",  default="cuda",
                   help="Device: cuda or cpu")
    p.add_argument("--workers", type=int, default=4,
                   help="DataLoader num_workers")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
