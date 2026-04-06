"""
GNN Local Game State Encoder.

Implements Section 3.3 of the paper (Equations 2 & 3):

  Node feature vector  x_i_t  (Eq. 2):
      [ pos_x, pos_y,          ← 2  (normalised position on pitch / screen)
        vel_x, vel_y,          ← 2  (normalised velocity)
        team_membership,       ← 1  (0 or 1)
        φ_X3D ]                ← D' (projected visual feature, D'=64)

  Edge Convolution update  (Eq. 3):
      h_{u}^{k+1} = MAX_{v ∈ N(u)}  MLP([ h_u^k | h_v^k - h_u^k ])

  Graph construction (Section 3.3.2):
      • One node per player per time step
      • Temporal edges: same player at adjacent time steps
      • Spatial edges: K-nearest neighbours in position space (K=6)

Outputs per-node embeddings h^{K} ∈ R^{B×N×T×GNN_OUT_DIM}

Dependencies:
    pip install torch torch-geometric
    (torch-geometric installation: https://pytorch-geometric.readthedocs.io)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import EdgeConv
    from torch_geometric.utils import knn_graph
    _PYGEOM_AVAILABLE = True
except ImportError:
    _PYGEOM_AVAILABLE = False
    print(
        "[GNN] torch_geometric not found — using manual EdgeConv fallback. "
        "Install with: pip install torch-geometric"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Manual EdgeConv (fallback when torch-geometric is unavailable)
# ──────────────────────────────────────────────────────────────────────────────
class EdgeConvManual(nn.Module):
    """
    Implements Equation 3:
        h_u^{k+1} = MAX_{v ∈ N(u)}  MLP(concat(h_u, h_v - h_u))

    Operates on a dense adjacency matrix rather than a sparse graph.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        h: torch.Tensor,          # (M, in_dim)   M = B*T*N nodes
        adj: torch.Tensor,        # (M, K)        indices of K neighbours
    ) -> torch.Tensor:
        """
        Args:
            h   : node features, shape (M, in_dim)
            adj : for each node, indices of its K neighbours, shape (M, K)
        Returns:
            h_new: (M, out_dim)
        """
        M, K = adj.shape
        in_dim = h.shape[-1]

        # Gather neighbour features: (M, K, in_dim)
        h_nbr = h[adj.view(-1)].view(M, K, in_dim)

        # Repeat central node K times: (M, K, in_dim)
        h_ctr = h.unsqueeze(1).expand(M, K, in_dim)

        # Edge features: concat(h_u, h_v - h_u) → (M, K, 2*in_dim)
        edge_feat = torch.cat([h_ctr, h_nbr - h_ctr], dim=-1)

        # Apply MLP per edge — reshape to (M*K, 2*in_dim)
        flat = edge_feat.view(M * K, 2 * in_dim)
        flat = self.mlp(flat)               # (M*K, out_dim)
        flat = flat.view(M, K, -1)          # (M, K, out_dim)

        # Max-pool over neighbours (channel-wise MAX, Equation 3)
        h_new = flat.max(dim=1).values      # (M, out_dim)
        return h_new


# ──────────────────────────────────────────────────────────────────────────────
# KNN helper (pure PyTorch, no torch-geometric required)
# ──────────────────────────────────────────────────────────────────────────────
def knn_indices(pos: torch.Tensor, k: int) -> torch.Tensor:
    """
    Compute K-nearest neighbour indices using Euclidean distance.

    Args:
        pos : (N, 2)  — x,y positions
        k   : number of neighbours (excluding self)
    Returns:
        idx : (N, K)  — neighbour indices  (self is excluded)
    """
    N = pos.shape[0]
    # Pairwise squared distances
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)   # N×N×2
    dist = (diff ** 2).sum(-1)                   # N×N

    # Exclude self by setting diagonal to a large value
    dist.fill_diagonal_(float("inf"))

    k_actual = min(k, N - 1)
    _, idx = dist.topk(k_actual, dim=-1, largest=False)  # N×k_actual

    # If N < k+1, pad with self-index
    if k_actual < k:
        pad = torch.arange(N, device=pos.device).unsqueeze(-1)
        pad = pad.expand(N, k - k_actual)
        idx = torch.cat([idx, pad], dim=-1)

    return idx   # N×K


# ──────────────────────────────────────────────────────────────────────────────
# Visual feature projector  (D=192 → D'=64)  used inside node features
# ──────────────────────────────────────────────────────────────────────────────
class VisualProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ──────────────────────────────────────────────────────────────────────────────
# Main GNN module
# ──────────────────────────────────────────────────────────────────────────────
class LocalGameStateGNN(nn.Module):
    """
    Encodes the local game state for each player at each time step.

    Pipeline:
        1. Project visual features Φ_X3D (D=192) → φ_X3D (D'=64)
        2. Build node feature vector x_i_t  (Equation 2)
        3. Build spatio-temporal graph G = (V, E)
        4. Apply GNN_LAYERS EdgeConv layers  (Equation 3)
        5. Return node embeddings h^K ∈ R^{B×N×T×GNN_OUT_DIM}

    Args:
        cfg : configuration object (configs/config.py)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Visual projector: D → D'
        self.vis_proj = VisualProjector(cfg.X3D_FEAT_DIM, cfg.VISUAL_PROJ_DIM)

        # Input to first EdgeConv layer = NODE_FEAT_DIM = 5 + D'
        in_dims  = [cfg.NODE_FEAT_DIM] + [cfg.GNN_HIDDEN_DIM] * (cfg.GNN_LAYERS - 1)
        out_dims = [cfg.GNN_HIDDEN_DIM] * (cfg.GNN_LAYERS - 1) + [cfg.GNN_OUT_DIM]

        self.edge_convs = nn.ModuleList([
            EdgeConvManual(in_dim=i, out_dim=o)
            for i, o in zip(in_dims, out_dims)
        ])

        self.k = cfg.K_NEIGHBORS
        self.out_dim = cfg.GNN_OUT_DIM

    # ── graph construction ────────────────────────────────────────────────────
    def _build_adjacency(
        self,
        pos: torch.Tensor,           # (N, 2)  positions at one time step
        player_mask: torch.Tensor,   # (N,)    True for valid players
    ) -> torch.Tensor:
        """
        Returns KNN adjacency indices (N, K).
        Invalid players are connected only to themselves.
        """
        valid_idx = player_mask.nonzero(as_tuple=True)[0]
        N = pos.shape[0]

        adj = torch.arange(N, device=pos.device).unsqueeze(-1).expand(N, self.k)
        adj = adj.clone()

        if valid_idx.numel() > 1:
            valid_pos = pos[valid_idx]
            valid_knn = knn_indices(valid_pos, self.k)  # (|valid|, K)
            # Remap local indices back to global
            adj[valid_idx] = valid_idx[valid_knn]

        return adj   # (N, K)

    # ── forward ───────────────────────────────────────────────────────────────
    def forward(
        self,
        phi_x3d: torch.Tensor,      # B×N×T×D  visual features
        positions: torch.Tensor,    # B×N×T×2  normalised positions (x,y)
        velocities: torch.Tensor,   # B×N×T×2  normalised velocities
        team_ids: torch.Tensor,     # B×N×T    0 or 1  (team membership)
        player_mask: torch.Tensor,  # B×N      True for valid players
    ) -> torch.Tensor:
        """
        Returns:
            h_K : B×N×T×GNN_OUT_DIM   node embeddings after K EdgeConv layers
        """
        B, N, T, D = phi_x3d.shape

        # 1. Project visual features → φ_X3D  (D → D')
        phi_proj = self.vis_proj(phi_x3d.view(B * N * T, D))   # B*N*T × D'
        phi_proj = phi_proj.view(B, N, T, self.cfg.VISUAL_PROJ_DIM)

        # 2. Build node feature vectors x_i_t  (Equation 2)
        #    [pos_x, pos_y, vel_x, vel_y, team, phi_proj]
        team_f = team_ids.float().unsqueeze(-1)   # B×N×T×1
        x = torch.cat([positions, velocities, team_f, phi_proj], dim=-1)
        # x : B×N×T×NODE_FEAT_DIM

        # 3. Process each (batch, time) slice through the GNN
        #    We flatten to M = B*T*N nodes, build adjacency per batch-time slice,
        #    then apply EdgeConv layers.

        # Reshape to (B*T, N, feat)
        x_bt = x.permute(0, 2, 1, 3).contiguous().view(B * T, N, -1)
        pos_bt = positions.permute(0, 2, 1, 3).contiguous().view(B * T, N, 2)
        mask_bt = player_mask.unsqueeze(1).expand(B, T, N).contiguous().view(B * T, N)

        # Build adjacency for each batch-time slice: list of (N, K) tensors
        # Then flatten all nodes to a single (B*T*N, feat) tensor with
        # a matching adjacency offset for each slice.
        BT = B * T
        all_adj = []
        for bt in range(BT):
            adj = self._build_adjacency(pos_bt[bt], mask_bt[bt])  # (N, K)
            adj = adj + bt * N   # global offset
            all_adj.append(adj)

        adj_global = torch.cat(all_adj, dim=0)  # (BT*N, K)

        # Flatten node features: (BT*N, feat)
        h = x_bt.view(BT * N, -1)

        # 4. Apply EdgeConv layers
        for layer in self.edge_convs:
            h = layer(h, adj_global)     # (BT*N, out_dim at this layer)

        # 5. Reshape back to B×N×T×GNN_OUT_DIM
        h = h.view(B, T, N, self.out_dim)
        h = h.permute(0, 2, 1, 3).contiguous()   # B×N×T×GNN_OUT_DIM

        return h
