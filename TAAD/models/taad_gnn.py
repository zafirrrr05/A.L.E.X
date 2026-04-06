"""
TAAD + GNN  —  Full Action Detection Model.

Combines (Figure 1 / Section 3):
    ┌─────────────────────────────────────────────────────────────────┐
    │  clip  B×3×T×H×W                                                │
    │    └── X3D + FPN + ROI Align ──► Φ_X3D   B×N×T×192              │
    │                                       │                         │
    │  game state (pos, vel, team)          │                         │
    │    └── GNN (EdgeConv × L) ──────────► h^K   B×N×T×128           │
    │                                       │                         │
    │  concat [ Φ_X3D | h^K ]  ──────────► combined  B×N×T×320        │
    │    └── TCN + MLP ──────────────────► logits   B×N×T×C           │
    └─────────────────────────────────────────────────────────────────┘

Training loss: cross-entropy per frame per player (Section 3.4)

Dependencies:
    pip install torch torchvision pytorchvideo torch-geometric
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .visual_extractor import VisualFeatureExtractor
from .gnn_game_state   import LocalGameStateGNN
from .tcn_head         import TCNHead


class TAADWithGNN(nn.Module):
    """
    Full TAAD + GNN model.

    Args:
        cfg         : config object  (configs/config.py)
        num_classes : total number of action classes (including background)
    """

    def __init__(self, cfg, num_classes: int):
        super().__init__()
        self.cfg         = cfg
        self.num_classes = num_classes

        # ── Sub-modules ──────────────────────────────────────────────────────
        self.visual_extractor = VisualFeatureExtractor(cfg)
        self.gnn              = LocalGameStateGNN(cfg)
        self.tcn_head         = TCNHead(cfg, num_classes)

        # Cross-entropy loss (Section 3.4)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    # ── forward ───────────────────────────────────────────────────────────────
    def forward(
        self,
        clip: torch.Tensor,           # B×3×T×H×W
        bboxes: torch.Tensor,         # B×N×T×4  — normalised [0,1]
        positions: torch.Tensor,      # B×N×T×2  — normalised positions
        velocities: torch.Tensor,     # B×N×T×2  — normalised velocities
        team_ids: torch.Tensor,       # B×N×T    — int64, 0 or 1
        player_mask: torch.Tensor,    # B×N      — bool
        labels: torch.Tensor = None,  # B×N×T    — int64, class index (-1=ignore)
    ) -> dict:
        """
        Returns a dict with:
            logits      : B×N×T×num_classes   (raw scores)
            probs       : B×N×T×num_classes   (softmax probabilities)
            loss        : scalar tensor  (only when labels is not None)
        """
        B, N, T, _ = bboxes.shape

        # ── 1. Visual features Φ_X3D  (Equation 1) ──────────────────────────
        phi_x3d = self.visual_extractor(clip, bboxes)   # B×N×T×192

        # ── 2. GNN game-state embeddings h^K  (Equations 2 & 3) ─────────────
        h_k = self.gnn(
            phi_x3d   = phi_x3d,
            positions  = positions,
            velocities = velocities,
            team_ids   = team_ids,
            player_mask= player_mask,
        )   # B×N×T×128

        # ── 3. Concatenate and feed to TCN head (Section 3.4) ────────────────
        combined = torch.cat([phi_x3d, h_k], dim=-1)   # B×N×T×320
        logits   = self.tcn_head(combined)              # B×N×T×num_classes

        probs = F.softmax(logits, dim=-1)

        out = {"logits": logits, "probs": probs}

        # ── 4. Loss ──────────────────────────────────────────────────────────
        if labels is not None:
            # Reshape to (B*N*T,) for cross-entropy
            logits_flat = logits.view(B * N * T, self.num_classes)
            labels_flat = labels.view(B * N * T)
            out["loss"] = self.loss_fn(logits_flat, labels_flat)

        return out

    # ── convenience: parameter groups for optimiser (Section 3.6) ────────────
    def get_param_groups(self, weight_decay: float):
        """
        Returns two parameter groups:
            - no-decay : bias and BatchNorm parameters
            - decay    : everything else
        Applied as in the paper: weight_decay only on non-bias params.
        """
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "bn" in name or "norm" in name:
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {"params": decay,    "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
