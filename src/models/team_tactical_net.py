import torch
import torch.nn as nn

class NodeEmbedding(nn.Module):
    def __init__(self, in_dim=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )

    def forward(self, x):
        # x: [B, N, 4]
        return self.net(x)
    
    
class SpatialInteractionBlock(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            dim, heads, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, N, D]
        h, _ = self.attn(x, x, x)
        x = self.norm1(x + h)
        h = self.ff(x)
        x = self.norm2(x + h)
        return x


class SpaceControlHead(nn.Module):
    def __init__(self, d_model, H=24, W=16):
        super().__init__()

        self.H = H
        self.W = W

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, H * W)
        )

    def forward(self, frame_tokens):
        # frame_tokens: [B,T,D]
        B, T, D = frame_tokens.shape

        out = self.mlp(frame_tokens)   # [B,T,H*W]
        out = out.view(B, T, self.H, self.W)

        return out


class FrameEncoder(nn.Module):
    def __init__(self, node_dim=64, layers=2):
        super().__init__()
        self.blocks = nn.ModuleList(
            [SpatialInteractionBlock(node_dim) for _ in range(layers)]
        )

    def forward(self, x):
        # x: [B, N, D]
        for b in self.blocks:
            x = b(x)
        return x

    
class FramePooling(nn.Module):
    def forward(self, x):
        # x: [B, N, D]
        return x.mean(dim=1)

    
class TemporalEncoder(nn.Module):
    def __init__(self, dim=64, depth=4, heads=4):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            batch_first=True
        )
        self.enc = nn.TransformerEncoder(layer, depth)

    def forward(self, x):
        # x: [B, T, D]
        return self.enc(x)


class TeamTacticalNet(nn.Module):

    def __init__(self, node_dim=64):
        super().__init__()

        self.node_embed = NodeEmbedding(4, node_dim)

        self.team1_encoder = FrameEncoder(node_dim, 2)
        self.team2_encoder = FrameEncoder(node_dim, 2)

        self.pool = FramePooling()

        # after pooling both teams + ball embedding
        self.fuse = nn.Linear(node_dim * 3, node_dim)

        self.temporal = TemporalEncoder(node_dim, 4, 4)

    def forward(self, team1, team2, ball):
        """
        team1: [B, T, 11, 4]
        team2: [B, T, 11, 4]
        ball:  [B, T, 4]
        """

        B, T, N, C = team1.shape

        # ----------------------------------
        # flatten time
        # ----------------------------------
        t1 = team1.view(B * T, N, C)
        t2 = team2.view(B * T, N, C)
        b  = ball.view(B * T, 1, C)

        # ----------------------------------
        # embeddings
        # ----------------------------------
        t1 = self.node_embed(t1)
        t2 = self.node_embed(t2)
        b  = self.node_embed(b)

        # ----------------------------------
        # spatial interaction (separate GNNs)
        # ----------------------------------
        t1 = self.team1_encoder(t1)
        t2 = self.team2_encoder(t2)

        # ----------------------------------
        # pool per frame
        # ----------------------------------
        t1_tok = self.pool(t1)   # [B*T, D]
        t2_tok = self.pool(t2)
        b_tok  = b.squeeze(1)

        # ----------------------------------
        # fuse
        # ----------------------------------
        frame_tok = torch.cat([t1_tok, t2_tok, b_tok], dim=-1)
        frame_tok = self.fuse(frame_tok)

        frame_tok = frame_tok.view(B, T, -1)

        # ----------------------------------
        # temporal encoder
        # ----------------------------------
        out = self.temporal(frame_tok)

        return out

class PassQualityHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.cls = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 2)
        )

    def forward(self, token):
        return self.cls(token)


class SpaceControlHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.reg = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

    def forward(self, tokens):
        # tokens: [B, T, D]
        return self.reg(tokens).squeeze(-1)


class MissedOpportunityHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.cls = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 2)
        )

    def forward(self, token):
        return self.cls(token)

class SubstitutionImpactHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.reg = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

    def forward(self, seq_token):
        return self.reg(seq_token)


class TacticalModel(nn.Module):

    def __init__(self, dim=64):
        super().__init__()

        self.backbone = TeamTacticalNet(dim)

        # existing heads (you already had)
        self.pass_head   = PassQualityHead(dim)
        self.space_head  = SpaceControlHead(dim)
        self.missed_head = MissedOpportunityHead(dim)
        self.sub_head    = SubstitutionImpactHead(dim)

        # -------------------------------------------------
        # SSL heads (NEW)
        # -------------------------------------------------

        # future task
        self.ball_head = nn.Linear(dim, 4)

        # masked players (22 players = 11 + 11)
        self.masked_head = nn.Linear(dim, 22 * 2)   # positions only

        # possession
        self.possession_head = nn.Linear(dim, 2)

        # temporal order
        self.order_head = nn.Linear(dim * 2, 2)

    def forward_backbone(self, team1, team2, ball):
        return self.backbone(team1, team2, ball)                                      