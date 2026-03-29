"""
dual_gatv2_model.py  (v2 — matches football_graph_dataset.py v2)
─────────────────────────────────────────────────────────────────
Hierarchical Dual GATv2 for football tactical analysis.

Graph 1 (player-level) :  23 nodes × 10 features,  4 edge features
Graph 2 (team-level)   :   3 nodes ×  7 features   (injected with G1 pool)
Temporal               :  Bi-LSTM over 50 frames with soft attention
Heads                  :  Formation · Pass network · Movement · Set piece

Install:
    pip install torch torch_geometric
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.data import Data

import numpy as np, tempfile, os
from src.training.graph_dataset import clip_to_graph_sequence
# ─────────────────────────────────────────────
#  GRAPH 1 — Player-level GATv2
# ─────────────────────────────────────────────

class PlayerLevelGATv2(nn.Module):
    """
    Two-layer GATv2 on the 23-node player graph.

    in_channels  : 10  (from engineer_player_features)
    edge_dim     : 4   (from build_edge_features)
    out per node : hidden_dim
    out graph    : 2 * hidden_dim  (mean-pool + max-pool concat)
    """

    def __init__(self,
                 in_channels: int   = 10,
                 edge_dim:    int   = 4,
                 hidden_dim:  int   = 64,
                 heads:       int   = 4,
                 dropout:     float = 0.1):
        super().__init__()

        self.conv1 = GATv2Conv(
            in_channels  = in_channels,
            out_channels = hidden_dim,
            heads        = heads,
            edge_dim     = edge_dim,
            dropout      = dropout,
            concat       = True,          # → heads * hidden_dim
        )
        self.norm1 = nn.LayerNorm(hidden_dim * heads)

        self.conv2 = GATv2Conv(
            in_channels  = hidden_dim * heads,
            out_channels = hidden_dim,
            heads        = 1,
            edge_dim     = edge_dim,
            dropout      = dropout,
            concat       = False,         # → hidden_dim
        )
        self.norm2   = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_node_dim  = hidden_dim
        self.out_graph_dim = hidden_dim * 2   # mean + max

    def forward(self, x, edge_index, edge_attr, batch=None):
        # layer 1
        h = self.conv1(x, edge_index, edge_attr)
        h = F.elu(self.norm1(h))
        h = self.dropout(h)

        # layer 2
        h = self.conv2(h, edge_index, edge_attr)
        h = F.elu(self.norm2(h))          # (N, hidden_dim)

        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        g = torch.cat([
            global_mean_pool(h, batch),   # (B, hidden_dim)
            global_max_pool(h,  batch),   # (B, hidden_dim)
        ], dim=-1)                        # (B, 2*hidden_dim)

        return h, g   # node embeddings, graph embedding


# ─────────────────────────────────────────────
#  GRAPH 2 — Team-level GATv2
# ─────────────────────────────────────────────

class TeamLevelGATv2(nn.Module):
    """
    One-layer GATv2 on the 3-node team graph.
    Node features are the pooled G1 embeddings injected per team.

    in_channels : must match the per-team embedding size fed in.
                  We project the raw 7-feature team stats up to this size
                  before passing in — see HierarchicalDualGATv2.forward().
    """

    def __init__(self,
                 in_channels: int   = 128,
                 hidden_dim:  int   = 64,
                 heads:       int   = 2,
                 dropout:     float = 0.1):
        super().__init__()

        self.conv = GATv2Conv(
            in_channels  = in_channels,
            out_channels = hidden_dim,
            heads        = heads,
            dropout      = dropout,
            concat       = False,
        )
        self.norm    = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden_dim

    def forward(self, x, edge_index):
        h = self.conv(x, edge_index)
        h = F.elu(self.norm(h))
        h = self.dropout(h)               # (3, hidden_dim)
        g = h.mean(dim=0, keepdim=True)  # (1, hidden_dim) — mean over 3 nodes
        return h, g


# ─────────────────────────────────────────────
#  TEMPORAL MODULE — Bi-LSTM with soft attention
# ─────────────────────────────────────────────

class TemporalBiLSTM(nn.Module):
    """
    Processes a sequence of per-frame embeddings (B, T, F) and returns
    a clip-level embedding (B, 2*hidden_dim) via Bi-LSTM + attention pool.
    """

    def __init__(self,
                 input_dim:  int   = 192,
                 hidden_dim: int   = 128,
                 num_layers: int   = 2,
                 dropout:    float = 0.1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size    = input_dim,
            hidden_size   = hidden_dim,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if num_layers > 1 else 0.0,
        )
        self.attn    = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden_dim * 2

    def forward(self, x_seq):
        """x_seq : (B, T, input_dim)  →  (B, 2*H)"""
        out, _ = self.lstm(x_seq)              # (B, T, 2*H)
        out    = self.dropout(out)
        scores  = self.attn(out).squeeze(-1)   # (B, T)
        weights = F.softmax(scores, dim=-1)    # (B, T)
        ctx     = (out * weights.unsqueeze(-1)).sum(dim=1)  # (B, 2*H)
        return ctx


# ─────────────────────────────────────────────
#  TASK HEADS
# ─────────────────────────────────────────────

def _mlp(in_dim: int, hidden: int, out_dim: int, dropout: float = 0.1):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.LayerNorm(hidden),
        nn.ELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    )


class TaskHeads(nn.Module):
    """
    Four heads sharing the clip embedding.

    formation_classes : e.g. 5  (4-4-2, 4-3-3, 3-5-2, 4-2-3-1, other)
    set_piece_classes : e.g. 4  (corner, free-kick, throw-in, open-play)
    pass_players      : 22      (one logit per player = most likely next passer)
    movement_dim      : 44      (22 players × Δx, Δy)
    """

    def __init__(self,
                 in_dim:             int = 256,
                 formation_classes:  int = 5,
                 set_piece_classes:  int = 4,
                 pass_players:       int = 22,
                 movement_dim:       int = 44,
                 dropout:            float = 0.1):
        super().__init__()
        self.formation    = _mlp(in_dim, 128, formation_classes,  dropout)
        self.set_piece    = _mlp(in_dim, 128, set_piece_classes,  dropout)
        self.pass_net     = _mlp(in_dim, 128, pass_players,       dropout)
        self.movement     = _mlp(in_dim, 128, movement_dim,       dropout)
        self.pass_quality = _mlp(in_dim,  64, 2,                  dropout)

    def forward(self, z):
        return {
            'formation':    self.formation(z),
            'set_piece':    self.set_piece(z),
            'pass_net':     self.pass_net(z),
            'movement':     self.movement(z),
            'pass_quality': self.pass_quality(z),
        }


# ─────────────────────────────────────────────
#  FULL MODEL
# ─────────────────────────────────────────────

class HierarchicalDualGATv2(nn.Module):
    """
    Full pipeline:

        .npz frame
            → PlayerLevelGATv2  (G1)   :  23 nodes, 10-feat, 4-feat edges
            → pool per team             :  mean of nodes 0-10, 11-21
            → project + concat stats    :  inject into team nodes
            → TeamLevelGATv2    (G2)   :  3 nodes
            → concat G1 + G2 per frame
            → TemporalBiLSTM           :  50 frames → clip embedding
            → TaskHeads                :  4 outputs
    """

    def __init__(self,
                 player_in:         int   = 10,
                 player_edge_dim:   int   = 4,
                 player_hidden:     int   = 64,
                 player_heads:      int   = 4,
                 team_stats_dim:    int   = 7,    # from frame_to_team_graph
                 team_hidden:       int   = 64,
                 lstm_hidden:       int   = 128,
                 formation_classes: int   = 5,
                 set_piece_classes: int   = 4,
                 dropout:           float = 0.1):
        super().__init__()

        # Graph 1
        self.player_gat = PlayerLevelGATv2(
            in_channels = player_in,
            edge_dim    = player_edge_dim,
            hidden_dim  = player_hidden,
            heads       = player_heads,
            dropout     = dropout,
        )
        g1_node_dim  = player_hidden           # per-node dim after G1
        g1_graph_dim = player_hidden * 2       # mean+max pooled

        # Project per-team pooled G1 embeddings → team-level in_channels
        # Each team node = mean of its 11 player G1 embeddings (g1_node_dim)
        # We project this to team_in_channels for TeamGAT
        team_in_channels = g1_node_dim         # 64
        self.team_gat = TeamLevelGATv2(
            in_channels = team_in_channels,
            hidden_dim  = team_hidden,
            dropout     = dropout,
        )
        g2_graph_dim = team_hidden

        # Per-frame concatenated embedding dimension
        frame_dim = g1_graph_dim + g2_graph_dim    # 128 + 64 = 192
        self.frame_proj = nn.Sequential(
            nn.Linear(frame_dim, frame_dim),
            nn.LayerNorm(frame_dim),
            nn.ELU(),
        )

        # Temporal
        self.temporal = TemporalBiLSTM(
            input_dim  = frame_dim,
            hidden_dim = lstm_hidden,
            dropout    = dropout,
        )
        lstm_out_dim = lstm_hidden * 2    # 256

        # Task heads
        self.heads = TaskHeads(
            in_dim            = lstm_out_dim,
            formation_classes = formation_classes,
            set_piece_classes = set_piece_classes,
            dropout           = dropout,
        )

    # ── single frame processing ───────────────

    def _process_frame(self, pg: Data, tg: Data) -> torch.Tensor:
        """
        Returns frame embedding of shape (1, frame_dim).
        """
        # ── G1: player-level ──
        node_embs, g1_emb = self.player_gat(
            pg.x,
            pg.edge_index,
            pg.edge_attr,
        )
        # node_embs : (23, g1_node_dim)
        # g1_emb    : (1,  g1_graph_dim)

        # ── Build team-level nodes from G1 node embeddings ──
        # Use only valid nodes for pooling (avoid padding-row bias)
        if hasattr(pg, 'valid_nodes'):
            valid = pg.valid_nodes  # (23,) bool
        else:
            valid = torch.ones(23, dtype=torch.bool, device=pg.x.device)

        valid_a = valid[:11]
        valid_b = valid[11:22]

        def safe_mean(embs, mask):
            if mask.any():
                return embs[mask].mean(0, keepdim=True)
            return torch.zeros(1, embs.shape[-1], device=embs.device)

        team_a_node = safe_mean(node_embs[:11],  valid_a)   # (1, g1_node_dim)
        team_b_node = safe_mean(node_embs[11:22], valid_b)  # (1, g1_node_dim)
        ball_node   = node_embs[22:23]                       # (1, g1_node_dim)

        team_x = torch.cat([team_a_node, team_b_node, ball_node], dim=0)  # (3, g1_node_dim)

        # ── G2: team-level ──
        _, g2_emb = self.team_gat(team_x, tg.edge_index)
        # g2_emb : (1, g2_out_dim)

        # ── concat & project ──
        frame_emb = torch.cat([g1_emb, g2_emb], dim=-1)    # (1, frame_dim)
        frame_emb = self.frame_proj(frame_emb)
        return frame_emb

    # ── forward ──────────────────────────────

    def forward(self, clip_sequences: list) -> dict:
        """
        clip_sequences : list[list[(player_graph, team_graph)]]
                         Outer list = batch of clips (B clips)
                         Inner list = T frames per clip

        Returns : dict of task output tensors, each (B, *)
        """
        batch_seqs = []

        for clip in clip_sequences:
            frame_embs = []
            for pg, tg in clip:
                fe = self._process_frame(pg, tg)  # (1, frame_dim)
                frame_embs.append(fe)
            seq = torch.cat(frame_embs, dim=0)    # (T, frame_dim)
            batch_seqs.append(seq)

        x_seq = torch.stack(batch_seqs, dim=0)    # (B, T, frame_dim)
        z     = self.temporal(x_seq)              # (B, lstm_out_dim)
        return self.heads(z)


# ─────────────────────────────────────────────
#  LOSS
# ─────────────────────────────────────────────

class TacticalLoss(nn.Module):
    """
    Multi-task weighted loss.
    Set weight=0 for any task you don't yet have labels for.
    """

    def __init__(self,
                 w_formation: float = 1.0,
                 w_set_piece: float = 1.0,
                 w_pass:      float = 0.5,
                 w_movement:  float = 0.3):
        super().__init__()
        self.w   = dict(formation=w_formation,
                        set_piece=w_set_piece,
                        pass_net=w_pass,
                        movement=w_movement)
        self.ce  = nn.CrossEntropyLoss(ignore_index=-1)
        self.mse = nn.MSELoss()

    def forward(self, preds: dict, targets: dict) -> torch.Tensor:
        loss = torch.zeros(1, device=next(iter(preds.values())).device)

        for key in ('formation', 'set_piece', 'pass_net'):
            if key in targets and self.w[key] > 0:
                loss = loss + self.w[key] * self.ce(preds[key], targets[key])

        if 'movement' in targets and self.w['movement'] > 0:
            loss = loss + self.w['movement'] * self.mse(
                preds['movement'], targets['movement'])

        return loss


# ─────────────────────────────────────────────
#  TRAINING LOOP SKELETON
# ─────────────────────────────────────────────

def train_one_epoch(model, dataloader, optimizer, device='cpu'):
    model.train()
    criterion = TacticalLoss()
    epoch_loss = 0.0

    for batch in dataloader:
        # batch is a list of (sequence, start_frame) from FootballTrackingDataset
        optimizer.zero_grad()

        clips = []
        for seq, _ in batch:
            frames = [(pg.to(device), tg.to(device)) for pg, tg in seq]
            clips.append(frames)

        preds = model(clips)

        # ── targets: adapt to your label format ──
        targets = {}
        labels  = [clips[i][0][0].y for i in range(len(clips))
                   if clips[i][0][0].y is not None]
        if labels:
            targets['formation'] = torch.cat(labels).to(device)
            targets['set_piece'] = torch.cat(labels).to(device)  # swap for real labels

        loss = criterion(preds, targets)

        if loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / max(len(dataloader), 1)


# ─────────────────────────────────────────────
#  ENTRY POINT — dry run
# ─────────────────────────────────────────────

if __name__ == '__main__':

    print("Building model...")
    model = HierarchicalDualGATv2()
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters : {total:,}")

    # create a temp .npz matching SequenceBuilder output exactly
    tmp = tempfile.NamedTemporaryFile(suffix='.npz', delete=False)
    # simulate some zero-padded rows (realistic: 9 real players, 2 padded)
    t1 = np.random.randn(50, 11, 4).astype(np.float32)
    t1[:, 9:, :] = 0.0   # last 2 rows padded
    t2 = np.random.randn(50, 11, 4).astype(np.float32)
    np.savez(tmp.name,
             players_team1 = t1,
             players_team2 = t2,
             ball          = np.random.randn(50, 4).astype(np.float32),
             referee       = np.random.randn(50, 4).astype(np.float32),
             start_frame   = np.array(0))

    seq, sf = clip_to_graph_sequence(tmp.name)
    print(f"Clip loaded : {len(seq)} frames  |  start_frame={sf}")

    # check padding mask
    pg0, _ = seq[0]
    print(f"Valid nodes (frame 0): {pg0.valid_nodes.sum().item()} / 23")

    model.eval()
    with torch.no_grad():
        out = model([seq])

    print("\nOutput shapes:")
    for k, v in out.items():
        print(f"  {k:12s}: {tuple(v.shape)}")

    os.unlink(tmp.name)
    print("\nAll checks passed.")
