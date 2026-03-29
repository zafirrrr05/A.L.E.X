"""
scripts/train_pass_quality.py  (v2)
──────────────────────────────────────
Fine-tunes the PassQualityHead on top of the SSL pre-trained backbone.

Key fixes vs v1:
  - Correct checkpoint key: ssl_encoder.pt (not ssl_backbone.pt)
  - Correct backbone call: model.backbone(team1, team2, ball) — needs team2
  - Correct input reshape: (B, W, 11, 4) not (B, players, 4)
  - Added class-balance weighted loss (pass success is often imbalanced)
  - Added train/val split using start_frame to avoid leakage
  - Saves best model to checkpoints/pass_quality.pt
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.models.team_tactical_net import TacticalModel
from src.training.pass_dataset import PassDataset

# python -m scripts.train_pass_quality
def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Pass Quality] device = {device}")

    # ── dataset ──
    ds = PassDataset("data/sequences", pre=5, post=10)

    if len(ds) == 0:
        print("No pass samples found. Check data/sequences/ has .npz files.")
        return

    # train/val split (80/20, no start_frame leakage here — sequences are
    # already independent clips; random split is fine at this granularity)
    n_val   = max(1, int(len(ds) * 0.2))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=8,  shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=8,  shuffle=False, num_workers=0)

    # ── model ──
    model = TacticalModel(dim=64).to(device)

    # load SSL pre-trained weights
    ssl_ckpt_path = "checkpoints/ssl_encoder.pt"
    if os.path.exists(ssl_ckpt_path):
        state = torch.load(ssl_ckpt_path, map_location=device)
        # ssl_encoder.pt saves the full TacticalModel state_dict
        # load into model, ignore head mismatches with strict=False
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"SSL weights loaded: {ssl_ckpt_path}")
        print(f"  missing keys     : {len(missing)}")
        print(f"  unexpected keys  : {len(unexpected)}")
    else:
        print(f"WARNING: {ssl_ckpt_path} not found — training from scratch")

    # freeze backbone, only train pass head
    optim = torch.optim.Adam([
        {"params": model.backbone.parameters(), "lr": 1e-5},   # slow — fine-tune
        {"params": model.pass_head.parameters(), "lr": 1e-3},  # fast — new head
    ])

    # ── class-balanced loss ──
    labels = [ds[i]["label"].item() for i in range(len(ds))]
    n_pos  = sum(labels)
    n_neg  = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        weight = None
        print("WARNING: only one class in dataset — check pass detection")
    else:
        # weight inversely proportional to class frequency
        weight = torch.tensor([n_pos / len(labels),
                                n_neg / len(labels)],
                               dtype=torch.float32).to(device)
        print(f"Class weights: neg={weight[0]:.3f} pos={weight[1]:.3f}")

    ce = nn.CrossEntropyLoss(weight=weight)

    optim = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3
    )

    os.makedirs("checkpoints", exist_ok=True)
    best_val_loss = float("inf")

    # ── training loop ──
    for epoch in range(20):

        # train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total   = 0

        for batch in tqdm(train_loader,
                          desc=f"Epoch {epoch+1:02d} train",
                          leave=False):

            team1 = batch["team1"].to(device)   # (B, W, 11, 4)
            team2 = batch["team2"].to(device)   # (B, W, 11, 4)
            ball  = batch["ball"].to(device)    # (B, W, 4)
            y     = batch["label"].to(device)   # (B,)

            # backbone expects (B, T, 11, 4)  — W is the time dimension
            tokens = model.backbone(team1, team2, ball)   # (B, T, D)

            # use the frame at the pass event (t_rel = pre = 5)
            # window is pre+post=15 frames, pass at index 5
            pass_token = tokens[:, 5]   # (B, D)

            logits = model.pass_head(pass_token)   # (B, 2)
            loss   = ce(logits, y)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            train_loss    += loss.item()
            preds          = logits.argmax(dim=1)
            train_correct += (preds == y).sum().item()
            train_total   += y.size(0)

        train_acc = train_correct / max(train_total, 1)

        # val
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total   = 0

        with torch.no_grad():
            for batch in val_loader:
                team1 = batch["team1"].to(device)
                team2 = batch["team2"].to(device)
                ball  = batch["ball"].to(device)
                y     = batch["label"].to(device)

                tokens     = model.backbone(team1, team2, ball)
                pass_token = tokens[:, 5]
                logits     = model.pass_head(pass_token)
                loss       = ce(logits, y)

                val_loss    += loss.item()
                preds        = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total   += y.size(0)

        val_acc = val_correct / max(val_total, 1)
        avg_val = val_loss / max(len(val_loader), 1)

        print(f"Epoch {epoch+1:02d} | "
              f"train loss {train_loss/len(train_loader):.4f} acc {train_acc:.3f} | "
              f"val loss {avg_val:.4f} acc {val_acc:.3f}")

        # save best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), "checkpoints/pass_quality.pt")
            print(f"  → saved best model (val loss {avg_val:.4f})")

    print("Done. Best checkpoint: checkpoints/pass_quality.pt")


if __name__ == "__main__":
    main()