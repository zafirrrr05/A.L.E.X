"""
Inference Runner — Action Detection Module.

Loads a trained TAAD+GNN checkpoint, processes a video clip (or folder),
and emits all three output formats required by Part 5 of the pipeline:

  1. JSON event dict      (API-2 format — fed to Formation Module)
  2. Action tubes         (list of ActionTube objects)
  3. Per-frame class probs (numpy array N×T×C — for downstream modules)

Usage:
    python -m TAAD.infer \
        --ckpt   checkpoints/taad_gnn.pt \
        --video  path/to/clip.avi \
        [--output results/events.json] \
        [--device cuda]

Or import and call `run_inference()` directly from your pipeline.

Dependencies:
    pip install torch torchvision pytorchvideo torch-geometric opencv-python tqdm
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .configs.config   import (
    CLIP_FRAMES, IMG_H, IMG_W, MAX_PLAYERS, FRAME_RATE,
    CONF_THRESHOLD, USE_PROXY_POSITIONS,
)
from .configs.labels   import NUM_CLASSES, IDX_TO_LABEL
from .models.taad_gnn  import TAADWithGNN
from .utils.dataset    import load_video_frames, _dummy_bboxes, _dummy_player_mask
from .utils.game_state import extract_game_state
from .utils.action_tubes import (
    build_action_tubes, tubes_to_json, events_to_json,
)
from .configs import config as cfg_module


# ──────────────────────────────────────────────────────────────────────────────
# Model loader
# ──────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: torch.device) -> TAADWithGNN:
    model = TAADWithGNN(cfg=cfg_module, num_classes=NUM_CLASSES)
    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    print(f"[Infer] Loaded checkpoint: {ckpt_path} (epoch {ck.get('epoch','?')})")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Single-clip inference
# ──────────────────────────────────────────────────────────────────────────────

def run_inference(
    model: TAADWithGNN,
    video_path: str,
    device: torch.device,
    bboxes_np: Optional[np.ndarray] = None,    # N×T×4  normalised — real tracker
    player_mask_np: Optional[np.ndarray] = None, # N  bool
    conf_threshold: float = CONF_THRESHOLD,
    fps: float = float(FRAME_RATE),
) -> dict:
    """
    Run action detection on a single video clip.

    Args:
        model          : loaded TAADWithGNN
        video_path     : path to .avi clip
        device         : torch device
        bboxes_np      : (optional) real tracker bboxes N×T×4.
                         If None, dummy full-frame bbox is used.
        player_mask_np : (optional) real player mask N.
                         If None, only player 0 (full-frame) is marked valid.
        conf_threshold : minimum tube confidence to include in output
        fps            : frames-per-second of the source video

    Returns:
        dict with keys:
            "action_tubes"  : list[ActionTube]
            "event_dicts"   : list[EventDict]
            "frame_probs"   : np.ndarray  N×T×C
            "event_json"    : str   (API-2 JSON string)
            "tubes_json"    : str
    """
    # ── 1. Load video ─────────────────────────────────────────────────────────
    frames = load_video_frames(video_path)
    if frames is None:
        raise ValueError(f"Cannot open video: {video_path}")

    T = CLIP_FRAMES

    # (T,H,W,3) → (1, 3, T, H, W)
    clip = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0).to(device)

    # ── 2. Bboxes ─────────────────────────────────────────────────────────────
    if bboxes_np is None:
        bboxes_np = _dummy_bboxes(T)          # N×T×4
    if player_mask_np is None:
        player_mask_np = _dummy_player_mask() # N

    bboxes      = torch.from_numpy(bboxes_np).unsqueeze(0).to(device)      # 1×N×T×4
    player_mask = torch.from_numpy(player_mask_np).unsqueeze(0).to(device) # 1×N

    # ── 3. Game state ─────────────────────────────────────────────────────────
    gs = extract_game_state(
        bboxes, player_mask=player_mask,
        use_proxy=USE_PROXY_POSITIONS,
    )

    # ── 4. Forward pass ───────────────────────────────────────────────────────
    with torch.no_grad():
        out = model(
            clip        = clip,
            bboxes      = bboxes,
            positions   = gs["positions"].to(device),
            velocities  = gs["velocities"].to(device),
            team_ids    = gs["team_ids"].to(device),
            player_mask = gs["player_mask"].to(device),
        )

    probs    = out["probs"]   # 1×N×T×C
    team_ids = gs["team_ids"] # 1×N×T

    # ── 5. Build outputs ──────────────────────────────────────────────────────
    results = build_action_tubes(
        probs         = probs,
        team_ids      = team_ids,
        fps           = fps,
        conf_threshold= conf_threshold,
        batch_idx     = 0,
    )

    results["event_json"] = events_to_json(results["event_dicts"])
    results["tubes_json"] = tubes_to_json(results["action_tubes"])

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline-compatible wrapper
# ──────────────────────────────────────────────────────────────────────────────

class ActionDetectionModule:
    """
    Drop-in module for the Part-5 pipeline (Section 5.3).

    Usage:
        detector = ActionDetectionModule("checkpoints/taad_gnn.pt")
        output   = detector.process("path/to/clip.avi")

        # Feed output to Formation Module:
        formation_input = output["event_dicts"]    # list[EventDict]
        frame_probs     = output["frame_probs"]    # N×T×C  numpy array
    """

    def __init__(self, ckpt_path: str, device: str = "cuda"):
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        self.model = load_model(ckpt_path, self.device)

    def process(
        self,
        video_path: str,
        bboxes: Optional[np.ndarray] = None,
        player_mask: Optional[np.ndarray] = None,
        conf_threshold: float = CONF_THRESHOLD,
    ) -> dict:
        """
        Process one clip.  Returns the three output formats.
        """
        return run_inference(
            model         = self.model,
            video_path    = video_path,
            device        = self.device,
            bboxes_np     = bboxes,
            player_mask_np= player_mask,
            conf_threshold= conf_threshold,
        )

    def process_and_emit_api2(self, video_path: str) -> str:
        """
        Convenience method: returns the API-2 JSON string directly.
        This is what gets passed to Module 2 (Formation) in Part 5 §5.10.
        """
        results = self.process(video_path)
        return results["event_json"]


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Run TAAD+GNN action detection")
    p.add_argument("--ckpt",   required=True, help="Checkpoint path")
    p.add_argument("--video",  required=True, help="Input video clip (.avi)")
    p.add_argument("--output", default=None,  help="Output JSON path (optional)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--conf",   type=float, default=CONF_THRESHOLD,
                   help="Confidence threshold")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model  = load_model(args.ckpt, device)

    results = run_inference(
        model         = model,
        video_path    = args.video,
        device        = device,
        conf_threshold= args.conf,
    )

    print("\n── Action Tubes ─────────────────────────────────────")
    print(results["tubes_json"])

    print("\n── API-2 Events (→ Formation Module) ────────────────")
    print(results["event_json"])

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(results["event_json"])
        print(f"\n✓ Events saved to {args.output}")
