"""
scripts/train_formation.py  (v2 — team1 only, quality filter)
───────────────────────────────────────────────────────────────
Trains the formation classification head of HierarchicalDualGATv2.

Changes vs v1:
  - Uses label_t1 only (label_t2 removed — team2 has insufficient coverage)
  - Quality filter is handled inside FormationDataset automatically
  - FormationGraphDataset only appends one sample per clip (not two)
  - max_clips applies to the raw FormationDataset before graph conversion

Labels are generated geometrically — no manual annotation needed.
Saves: checkpoints/formation_gatv2.pt
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.models.dual_gatv2_model import HierarchicalDualGATv2
from src.training.graph_dataset import frame_to_player_graph, frame_to_team_graph
from src.training.formation_analysis import (
    FormationDataset,
    FORMATION_CLASSES,
    FORMATION_NAMES,
)


# ─────────────────────────────────────────────
#  GRAPH FORMATION DATASET
# ─────────────────────────────────────────────

class FormationGraphDataset(torch.utils.data.Dataset):
    """
    Wraps FormationDataset and converts each clip to a PyG graph sequence.
    One sample per clip (team1 label only).

    Each sample:
        sequence : list of (player_graph, team_graph) — 50 frames
        label    : int  formation class (0-4)
    """

    def __init__(self, root_dir: str, max_clips: int = None):
        self.samples = []

        # FormationDataset already applies quality filter internally
        raw = FormationDataset(root_dir)

        clips_to_use = raw.samples
        if max_clips is not None:
            clips_to_use = clips_to_use[:max_clips]

        print(f"Converting {len(clips_to_use)} clips to graph sequences...")

        for item in tqdm(clips_to_use, desc="Building graph sequences"):
            t1   = item["team1"].numpy()   # (T, 11, 4) normalised
            t2   = item["team2"].numpy()
            ball = item["ball"].numpy()
            T    = t1.shape[0]

            sequence = []
            for f in range(T):
                pg = frame_to_player_graph(t1[f], t2[f], ball[f])
                tg = frame_to_team_graph(t1[f],  t2[f], ball[f])
                sequence.append((pg, tg))

            # one sample per clip — team1 label only
            self.samples.append({
                "sequence": sequence,
                "label":    item["label_t1"],   # ← team1 only
            })

        print(f"FormationGraphDataset: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        return {
            "sequence": s["sequence"],
            "label":    torch.tensor(s["label"], dtype=torch.long),
        }


# ─────────────────────────────────────────────
#  COLLATE
# ─────────────────────────────────────────────

def formation_collate(batch):
    sequences = [item["sequence"] for item in batch]
    labels    = torch.stack([item["label"] for item in batch])
    return sequences, labels


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Formation GATv2] device = {device}")

    # ── dataset ──
    # max_clips=5000 gives ~5000 samples after filtering
    # increase for better accuracy once you confirm training works
    ds = FormationGraphDataset("data/sequences", max_clips=5000)

    if len(ds) == 0:
        print("No samples found. Check data/sequences/ and quality filter.")
        return

    n_val   = max(1, int(len(ds) * 0.2))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              collate_fn=formation_collate, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              collate_fn=formation_collate, num_workers=0)

    # ── model ──
    model = HierarchicalDualGATv2(
        player_in         = 10,
        player_edge_dim   = 4,
        player_hidden     = 64,
        player_heads      = 4,
        team_hidden       = 64,
        lstm_hidden       = 128,
        formation_classes = 5,
        set_piece_classes = 4,
    ).to(device)

    # load pass quality weights if available — partial transfer
    ckpt_path = "checkpoints/pass_gatv2.pt"
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"Loaded weights from {ckpt_path}")
    else:
        print("Training from scratch")

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total:,}")

    # ── class distribution and weights ──
    labels_all   = [ds[i]["label"].item() for i in range(len(ds))]
    class_counts = {}
    for lbl in labels_all:
        class_counts[lbl] = class_counts.get(lbl, 0) + 1

    print("Formation distribution:")
    for cls in range(5):
        count = class_counts.get(cls, 0)
        print(f"  {FORMATION_NAMES.get(cls,'other'):12s} ({cls}): {count:5d}")

    total_samples = len(labels_all)
    weights = torch.zeros(5)
    for cls in range(5):
        count = class_counts.get(cls, 0)
        weights[cls] = total_samples / max(count * 5, 1)
    weights = weights.to(device)

    ce = nn.CrossEntropyLoss(weight=weights)

    # two-speed optimizer
    optim = torch.optim.Adam([
        {"params": model.player_gat.parameters(), "lr": 1e-4},
        {"params": model.team_gat.parameters(),   "lr": 1e-4},
        {"params": model.temporal.parameters(),   "lr": 5e-5},
        {"params": model.heads.parameters(),      "lr": 5e-4},
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optim, step_size=5, gamma=0.5
    )

    os.makedirs("checkpoints", exist_ok=True)
    best_val    = float("inf")
    accum_steps = 8

    for epoch in range(20):

        # ── train ──
        model.train()
        train_loss    = 0.0
        train_correct = 0
        train_total   = 0
        optim.zero_grad()

        for step, (sequences, labels) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1:02d} train", leave=False)
        ):
            labels = labels.to(device)

            clips = [[(pg.to(device), tg.to(device)) for pg, tg in seq]
                     for seq in sequences]

            out    = model(clips)
            logits = out["formation"]          # (1, 5)
            loss   = ce(logits, labels) / accum_steps
            loss.backward()

            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                optim.zero_grad()

            train_loss    += loss.item() * accum_steps
            preds          = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += labels.size(0)

        train_acc = train_correct / max(train_total, 1)
        scheduler.step()

        # ── val ──
        model.eval()
        val_loss    = 0.0
        val_correct = 0
        val_total   = 0

        with torch.no_grad():
            for sequences, labels in val_loader:
                labels = labels.to(device)
                clips  = [[(pg.to(device), tg.to(device)) for pg, tg in seq]
                          for seq in sequences]
                out    = model(clips)
                logits = out["formation"]
                loss   = ce(logits, labels)
                val_loss    += loss.item()
                preds        = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        avg_val = val_loss / max(len(val_loader), 1)
        val_acc = val_correct / max(val_total, 1)

        print(f"Epoch {epoch+1:02d} | "
              f"train loss {train_loss/len(train_loader):.4f} "
              f"acc {train_acc:.3f} | "
              f"val loss {avg_val:.4f} acc {val_acc:.3f}")

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(),
                       "checkpoints/formation_gatv2.pt")
            print(f"  → saved (val loss {avg_val:.4f})")

    print("Done. Best checkpoint: checkpoints/formation_gatv2.pt")


if __name__ == "__main__":
    main()