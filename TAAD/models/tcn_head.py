"""
Temporal Convolutional Network (TCN) — Section 3.2.1 / TAAD.

Takes the concatenated visual + GNN features:
    Φ_combined ∈ R^{B×N×T×(D + GNN_OUT_DIM)}   = R^{B×N×T×320}

and produces dense per-frame action predictions:
    logits ∈ R^{B×N×T×NUM_CLASSES}

The TCN slides a 1-D convolution along the temporal axis T for each player
independently.  A final MLP maps to class logits.

Architecture (Section 3.4):
    concat(Φ_X3D, h^K)  →  [TCN block × L]  →  MLP  →  logits

Dependencies:
    pip install torch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Single TCN block: temporal conv + BN + ReLU + dropout + residual
# ──────────────────────────────────────────────────────────────────────────────
class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, dropout: float = 0.3):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel,
                              padding=pad, bias=False)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        # Residual projection if channel dims differ
        self.res  = (nn.Conv1d(in_ch, out_ch, 1, bias=False)
                     if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (M, C, T)"""
        out = self.drop(F.relu(self.bn(self.conv(x)), inplace=True))
        return out + self.res(x)


# ──────────────────────────────────────────────────────────────────────────────
# Full TCN head
# ──────────────────────────────────────────────────────────────────────────────
class TCNHead(nn.Module):
    """
    Temporal Convolutional Network applied per player along the time axis.

    Input  : combined_feat  B×N×T×combined_dim   (visual concat GNN)
    Output : logits         B×N×T×num_classes
    """

    def __init__(self, cfg, num_classes: int):
        super().__init__()
        self.cfg = cfg

        channels = [cfg.COMBINED_DIM] + cfg.TCN_CHANNELS
        blocks = []
        for i in range(len(cfg.TCN_CHANNELS)):
            blocks.append(
                TCNBlock(
                    in_ch=channels[i],
                    out_ch=channels[i + 1],
                    kernel=cfg.TCN_KERNEL,
                    dropout=cfg.TCN_DROPOUT,
                )
            )
        self.tcn = nn.Sequential(*blocks)

        # MLP head: last TCN channel → num_classes  (Section 3.4)
        last_ch = channels[-1]
        self.mlp = nn.Sequential(
            nn.Linear(last_ch, last_ch // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.TCN_DROPOUT),
            nn.Linear(last_ch // 2, num_classes),
        )

    def forward(self, combined_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            combined_feat: B×N×T×combined_dim
        Returns:
            logits: B×N×T×num_classes
        """
        B, N, T, C = combined_feat.shape

        # Merge batch + player dimensions → (B*N, C, T)  for 1-D conv along T
        x = combined_feat.view(B * N, T, C).permute(0, 2, 1)   # B*N × C × T

        x = self.tcn(x)   # B*N × last_ch × T

        # Back to (B*N, T, last_ch) then MLP per frame
        x = x.permute(0, 2, 1)   # B*N × T × last_ch
        logits = self.mlp(x)      # B*N × T × num_classes

        logits = logits.view(B, N, T, -1)   # B×N×T×num_classes
        return logits
