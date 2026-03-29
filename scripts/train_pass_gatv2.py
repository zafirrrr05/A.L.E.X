"""
scripts/train_pass_gatv2.py
────────────────────────────
Trains the pass quality head of HierarchicalDualGATv2.

Pipeline:
    PassGraphDataset  →  HierarchicalDualGATv2  →  pass_net head  →  CE loss

The GATv2 cross-team edges explicitly encode defensive pressure and
receiver space — directly matching the geometric labels.

SSL pre-trained weights from checkpoints/ssl_encoder.pt are NOT loaded
here because the backbone architecture is different (GATv2 vs Transformer).
The GATv2 trains from scratch but converges fast because:
  - The graph structure encodes inter-team relationships directly
  - The geometric labels are clean and consistent
  - 6000+ samples is sufficient for a ~450k param model

Saves: checkpoints/pass_gatv2.pt
"""

# python -m scripts.train_pass_gatv2
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.models.dual_gatv2_model import HierarchicalDualGATv2
from src.training.pass_graph_dataset import PassGraphDataset


# ─────────────────────────────────────────────
#  COLLATE  — handles variable-length graph sequences
# ─────────────────────────────────────────────

def pass_collate(batch):
    """
    batch : list of dicts with keys sequence, label, t_rel

    Returns:
        sequences : list of graph sequences (one per sample)
        labels    : (B,) long tensor
        t_rels    : list of ints
    """
    sequences = [item["sequence"] for item in batch]
    labels    = torch.stack([item["label"] for item in batch])
    t_rels    = [item["t_rel"] for item in batch]
    return sequences, labels, t_rels


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Pass GATv2] device = {device}")

    # ── dataset ──
    ds = PassGraphDataset("data/sequences", pre=5, post=10)

    if len(ds) == 0:
        print("No samples found. Check data/sequences/.")
        return

    n_val   = max(1, int(len(ds) * 0.2))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    # batch_size=1 because each sample is a graph sequence
    # (GATv2 processes variable graphs — batching requires PyG Batch,
    #  easier to keep batch=1 and accumulate gradients)
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        collate_fn=pass_collate,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=pass_collate,
        num_workers=0,
    )

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

    # ── class-balanced loss ──
    weight = torch.tensor([2.0, 1.0], dtype=torch.float32).to(device)
    print(f"Class weights: bad={weight[0]:.3f}  good={weight[1]:.3f}")
    ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.05)

    # ── optimizer ──
    # two-speed: GATv2 layers learn faster than LSTM temporal
    optim = torch.optim.AdamW([
        {"params": model.player_gat.parameters(), "lr": 1e-4, "weight_decay": 1e-3},
        {"params": model.team_gat.parameters(),   "lr": 1e-4, "weight_decay": 1e-3},
        {"params": model.temporal.parameters(),   "lr": 5e-5, "weight_decay": 1e-3},
        {"params": model.heads.parameters(), "lr": 5e-4, "weight_decay": 1e-4},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=20, eta_min=1e-6
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    os.makedirs("checkpoints", exist_ok=True)
    best_val_loss = float("inf")

    # ── gradient accumulation (simulates larger batch size) ──
    accum_steps = 8   # effective batch size = 8

    # ── training loop ──
    for epoch in range(20):

        # train
        model.train()
        train_loss    = 0.0
        train_correct = 0
        train_total   = 0

        optim.zero_grad()

        for step, (sequences, labels, t_rels) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1:02d} train", leave=False)
        ):
            labels = labels.to(device)

            # move graphs to device
            clips = []
            for seq in sequences:
                frames = [(pg.to(device), tg.to(device)) for pg, tg in seq]
                clips.append(frames)

            # forward — get all task outputs
            out    = model(clips)
            logits = out["pass_quality"]   # (1, 2) — direct binary output

            loss = ce(logits, labels) / accum_steps
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

        # val
        model.eval()
        val_loss    = 0.0
        val_correct = 0
        val_total   = 0

        with torch.no_grad():
            for sequences, labels, t_rels in val_loader:
                labels = labels.to(device)

                clips = []
                for seq in sequences:
                    frames = [(pg.to(device), tg.to(device)) for pg, tg in seq]
                    clips.append(frames)

                out    = model(clips)
                logits = out["pass_quality"]   # (1, 2)

                loss = ce(logits, labels)
                val_loss    += loss.item()
                preds        = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        avg_val  = val_loss / max(len(val_loader), 1)
        val_acc  = val_correct / max(val_total, 1)

        print(f"Epoch {epoch+1:02d} | "
              f"train loss {train_loss/len(train_loader):.4f} acc {train_acc:.3f} | "
              f"val loss {avg_val:.4f} acc {val_acc:.3f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), "checkpoints/pass_gatv2.pt")
            print(f"  → saved (val loss {avg_val:.4f})")

        if epoch > 8 and avg_val > best_val_loss + 0.05:
            print("Early stopping — val loss diverging")
            break
    print("Done. Best checkpoint: checkpoints/pass_gatv2.pt")


if __name__ == "__main__":
    main()