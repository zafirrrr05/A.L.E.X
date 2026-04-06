"""
X3D + FPN visual feature extractor.

Implements Section 3.2.2 of the paper:
  - X3D-S backbone (pre-trained on Kinetics-400)
  - Feature Pyramid Network on the last 3 blocks
  - ROI Align along player tracklets → per-player, per-frame feature vectors
    Φ_X3D ∈ R^{N×T×192}

The backbone preserves the temporal dimension so that a T-frame clip
produces a temporal feature dimension of T (no temporal sub-sampling).

Dependencies:
    pip install torch torchvision pytorchvideo torchvision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

try:
    from pytorchvideo.models.x3d import create_x3d
    _PTV_AVAILABLE = True
except ImportError:
    _PTV_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Feature Pyramid Network (1-D channel-only, applied per spatial feature map)
# ──────────────────────────────────────────────────────────────────────────────
class FPN(nn.Module):
    """
    Lightweight FPN that fuses the last 3 X3D block outputs.
    All feature maps are projected to `out_channels` and summed top-down.
    """

    def __init__(self, in_channels_list: list[int], out_channels: int = 192):
        super().__init__()
        self.lateral = nn.ModuleList([
            nn.Conv3d(c, out_channels, kernel_size=1, bias=False)
            for c in in_channels_list
        ])
        self.out_channels = out_channels

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: list of tensors [C3, C4, C5], shapes B×C×T×H×W
        Returns:
            fused: B×out_channels×T×H×W  (same spatial/temporal as C5)
        """
        assert len(features) == len(self.lateral)

        # Project each level
        laterals = [l(f) for l, f in zip(self.lateral, features)]

        # Top-down fusion: start from deepest level
        out = laterals[-1]
        for i in range(len(laterals) - 2, -1, -1):
            # Upsample to match current level's spatial size
            target = laterals[i]
            up = F.interpolate(
                out,
                size=target.shape[2:],   # (T, H, W)
                mode="nearest",
            )
            out = target + up

        # Return the fused map at the top resolution
        return out


# ──────────────────────────────────────────────────────────────────────────────
# X3D Backbone wrapper — extracts multi-scale features for FPN
# ──────────────────────────────────────────────────────────────────────────────
class X3DBackbone(nn.Module):
    """
    Wraps pytorchvideo's X3D-S and exposes the intermediate block outputs
    needed by the FPN (blocks 3, 4, 5 in 1-indexed terms).

    If pytorchvideo is unavailable, falls back to a lightweight 3D-CNN stub
    with the same output shapes for local development / unit-testing.
    """

    def __init__(self, model_name: str = "x3d_s", pretrained: bool = True):
        super().__init__()
        self.model_name = model_name

        if _PTV_AVAILABLE:
            self._build_from_pytorchvideo(model_name, pretrained)
        else:
            print(
                "[X3DBackbone] pytorchvideo not found — using lightweight stub. "
                "Install with: pip install pytorchvideo"
            )
            self._build_stub()

    # ── pytorchvideo path ────────────────────────────────────────────────────
    def _build_from_pytorchvideo(self, model_name: str, pretrained: bool):
        from pytorchvideo.models.hub import (
            x3d_s, x3d_m, x3d_xs, x3d_l
        )
        builders = {"x3d_s": x3d_s, "x3d_m": x3d_m,
                    "x3d_xs": x3d_xs, "x3d_l": x3d_l}
        full_model = builders[model_name](pretrained=pretrained)

        # X3D-S block structure: stem → res1 → res2 → res3 → res4 → head
        # We use blocks 1-5 (res1..res4 + the 5th block / head stem)
        # Expose blocks 3, 4, 5 to FPN.
        self.stem   = full_model.blocks[0]   # stem
        self.block1 = full_model.blocks[1]   # res1
        self.block2 = full_model.blocks[2]   # res2
        self.block3 = full_model.blocks[3]   # res3  → FPN level 0
        self.block4 = full_model.blocks[4]   # res4  → FPN level 1
        # block5 is the projection conv (X3D's "head" stem conv) → FPN level 2
        self.block5 = full_model.blocks[5].proj  # 1×1×1 conv to 192 channels

        # Output channels for FPN lateral connections
        # X3D-S: block3→96ch, block4→192ch, block5→192ch  (approx, model-specific)
        self.out_channels = [96, 192, 192]
        self._use_stub = False

    # ── stub path (no pytorchvideo) ──────────────────────────────────────────
    def _build_stub(self):
        """
        3-block 3D-CNN stub that produces the same channel counts and roughly
        the same spatial/temporal reduction as X3D-S.  Useful for debugging
        graph + TCN layers without the full pre-trained backbone.
        """
        def _block(cin, cout, stride=(1, 2, 2)):
            return nn.Sequential(
                nn.Conv3d(cin, cout, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm3d(cout),
                nn.ReLU(inplace=True),
            )

        self.stem   = nn.Sequential(
            nn.Conv3d(3, 24, kernel_size=(1, 3, 3), stride=(1, 2, 2),
                      padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(24), nn.ReLU(inplace=True),
        )
        self.block1 = _block(24,  48,  stride=(1, 1, 1))
        self.block2 = _block(48,  96,  stride=(1, 2, 2))
        self.block3 = _block(96,  96,  stride=(1, 1, 1))   # FPN lvl 0
        self.block4 = _block(96,  192, stride=(1, 2, 2))   # FPN lvl 1
        self.block5 = _block(192, 192, stride=(1, 1, 1))   # FPN lvl 2

        self.out_channels = [96, 192, 192]
        self._use_stub = True

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            x: B×3×T×H×W  (float32, normalised)
        Returns:
            [c3, c4, c5]: multi-scale feature maps for FPN
        """
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        c3 = self.block3(x)
        c4 = self.block4(c3)
        c5 = self.block5(c4)
        return [c3, c4, c5]


# ──────────────────────────────────────────────────────────────────────────────
# ROI Align along player tracklets
# ──────────────────────────────────────────────────────────────────────────────
def roi_align_tracklets(
    feat_map: torch.Tensor,          # B×C×T×H×W
    bboxes: torch.Tensor,            # B×N×T×4  (x1,y1,x2,y2 in [0,1] norm.)
    output_size: tuple[int, int] = (4, 4),
    spatial_scale: float = 1.0,
) -> torch.Tensor:
    """
    Applies ROI Align at every time step for every player.

    The feature map has already been temporally aligned with the clip
    (temporal dim == T), so we slice frame-by-frame.

    Args:
        feat_map      : B×C×T×H×W
        bboxes        : B×N×T×4  — normalised [0,1] bounding boxes
        output_size   : (rH, rW) — ROI Align output spatial size
        spatial_scale : scales from [0,1] coords to feature map coords;
                        pass 1.0 when bboxes are already scaled to feat_map.

    Returns:
        player_feats  : B×N×T×C×rH×rW
    """
    B, C, T, H, W = feat_map.shape
    _, N, _, _ = bboxes.shape
    rH, rW = output_size

    # Scale bboxes from [0,1] to feature-map pixel coords
    scale = torch.tensor([W, H, W, H], dtype=torch.float32,
                         device=feat_map.device)

    out = torch.zeros(B, N, T, C, rH, rW, device=feat_map.device,
                      dtype=feat_map.dtype)

    for t in range(T):
        frame_feat = feat_map[:, :, t, :, :]   # B×C×H×W

        # Build roi list: [[batch_idx, x1, y1, x2, y2], ...]
        rois = []
        for b in range(B):
            for n in range(N):
                box = bboxes[b, n, t] * scale      # absolute pixels in feat
                rois.append(torch.cat([
                    torch.tensor([b], dtype=torch.float32,
                                 device=feat_map.device),
                    box,
                ]))
        rois = torch.stack(rois, dim=0)            # (B*N)×5

        pooled = roi_align(
            frame_feat, rois,
            output_size=output_size,
            spatial_scale=spatial_scale,
            aligned=True,
        )  # (B*N)×C×rH×rW

        pooled = pooled.view(B, N, C, rH, rW)
        out[:, :, t, :, :, :] = pooled

    return out   # B×N×T×C×rH×rW


# ──────────────────────────────────────────────────────────────────────────────
# Full visual feature extractor (X3D + FPN + ROI Align → Φ_X3D)
# ──────────────────────────────────────────────────────────────────────────────
class VisualFeatureExtractor(nn.Module):
    """
    Produces per-player, per-frame visual features Φ_X3D ∈ R^{N×T×D}
    as described in Section 3.2.2 (Equation 1).

    Output D = cfg.X3D_FEAT_DIM = 192.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.backbone = X3DBackbone(
            model_name=cfg.X3D_MODEL, pretrained=True
        )
        self.fpn = FPN(
            in_channels_list=self.backbone.out_channels,
            out_channels=cfg.X3D_FEAT_DIM,
        )
        # Flatten spatial ROI pooling output  (C × rH × rW → C)
        rH, rW = cfg.ROI_OUTPUT_SIZE
        self.pool_flat = nn.AdaptiveAvgPool2d((1, 1))  # C×rH×rW → C×1×1

        self.out_dim = cfg.X3D_FEAT_DIM   # D = 192

    def forward(
        self,
        clip: torch.Tensor,        # B×3×T×H×W
        bboxes: torch.Tensor,      # B×N×T×4  (normalised [0,1])
    ) -> torch.Tensor:
        """
        Returns:
            phi: B×N×T×D   (Φ_X3D  from Equation 1)
        """
        B, N, T, _ = bboxes.shape

        # 1. Multi-scale backbone features
        c3, c4, c5 = self.backbone(clip)

        # 2. FPN → single fused feature map at top resolution
        fused = self.fpn([c3, c4, c5])  # B×D×T'×H'×W'

        # The backbone may reduce T slightly; interpolate back to T if needed
        if fused.shape[2] != T:
            fused = F.interpolate(
                fused, size=(T, fused.shape[3], fused.shape[4]),
                mode="trilinear", align_corners=False,
            )

        # 3. ROI Align along tracklets
        # Scale bboxes to fused feature map space
        roi_feats = roi_align_tracklets(
            fused, bboxes,
            output_size=self.cfg.ROI_OUTPUT_SIZE,
            spatial_scale=1.0,
        )  # B×N×T×D×rH×rW

        # 4. Flatten spatial dims
        B_, N_, T_, D_, rH, rW = roi_feats.shape
        roi_feats = roi_feats.view(B_ * N_ * T_, D_, rH, rW)
        roi_feats = self.pool_flat(roi_feats)          # B*N*T×D×1×1
        phi = roi_feats.view(B_, N_, T_, D_)           # B×N×T×D

        return phi   # Φ_X3D ∈ R^{B×N×T×192}
