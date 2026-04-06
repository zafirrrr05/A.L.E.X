"""
Evaluation — mean Average Precision (mAP) at temporal IoU thresholds.

Implements Section 4.1 of the paper:
  - Temporal IoU matching between predicted and ground-truth tubes
  - 11-point interpolated Average Precision per class  (VOC method)
  - mAP averaged across foreground classes

Also computes the True-Positive / False-Positive breakdown from
Section 4.3.1 (high recall - low precision analysis).

Usage:
    python -m TAAD.evaluate \
        --ckpt  checkpoints/taad_gnn.pt \
        --data  data/raw_videos/clip_dataset \
        [--iou_thresholds 0.2 0.5] \
        [--conf 0.5]

Dependencies:
    pip install torch torchvision pytorchvideo torch-geometric tqdm
"""

from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from .configs.config   import (
    CLIP_FRAMES, FRAME_RATE, CONF_THRESHOLD, USE_PROXY_POSITIONS,
)
from .configs.labels   import NUM_CLASSES, IDX_TO_LABEL, BACKGROUND
from .models.taad_gnn  import TAADWithGNN
from .utils.dataset    import build_dataloaders
from .utils.game_state import extract_game_state
from .utils.action_tubes import build_action_tubes, ActionTube
from .configs import config as cfg_module


# ──────────────────────────────────────────────────────────────────────────────
# Temporal IoU
# ──────────────────────────────────────────────────────────────────────────────

def temporal_iou(
    pred: tuple[int, int],   # (start_frame, end_frame)
    gt:   tuple[int, int],
) -> float:
    """Compute temporal intersection-over-union for two frame intervals."""
    ps, pe = pred
    gs, ge = gt
    inter  = max(0, min(pe, ge) - max(ps, gs) + 1)
    union  = (pe - ps + 1) + (ge - gs + 1) - inter
    return inter / union if union > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# 11-point interpolated Average Precision  (VOC 2010 method)
# ──────────────────────────────────────────────────────────────────────────────

def voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    """
    Compute 11-point interpolated AP as used in the paper (Section 4.1).
    """
    ap = 0.0
    for thr in np.linspace(0.0, 1.0, 11):
        p_at_r = prec[rec >= thr].max() if np.any(rec >= thr) else 0.0
        ap += p_at_r / 11.0
    return float(ap)


def compute_ap_for_class(
    preds: list[dict],   # [{"score": float, "matched": bool}, ...]
    n_gt:  int,
) -> float:
    """
    Compute AP for a single class given sorted predictions and #GT instances.
    """
    if n_gt == 0:
        return float("nan")

    preds_sorted = sorted(preds, key=lambda x: x["score"], reverse=True)
    tp = np.zeros(len(preds_sorted))
    fp = np.zeros(len(preds_sorted))

    for i, p in enumerate(preds_sorted):
        tp[i] = 1 if p["matched"] else 0
        fp[i] = 0 if p["matched"] else 1

    tp_cum  = np.cumsum(tp)
    fp_cum  = np.cumsum(fp)
    rec     = tp_cum / n_gt
    prec    = tp_cum / (tp_cum + fp_cum + 1e-9)

    return voc_ap(rec, prec)


# ──────────────────────────────────────────────────────────────────────────────
# Match predictions to GT tubes at a given IoU threshold
# ──────────────────────────────────────────────────────────────────────────────

def match_tubes(
    pred_tubes: list[ActionTube],
    gt_tubes:   list[ActionTube],
    iou_threshold: float = 0.2,
) -> tuple[list[dict], dict[int, int]]:
    """
    Match predicted to ground-truth tubes using temporal IoU.
    Returns:
        class_preds : per-class list of {"score", "matched"} dicts
        n_gt_per_cls: per-class count of ground-truth tubes
    """
    class_preds: dict[int, list] = defaultdict(list)
    n_gt_per_cls: dict[int, int] = defaultdict(int)

    # Count GT per class
    for gt in gt_tubes:
        n_gt_per_cls[gt.class_idx] += 1

    # Sort predictions by score descending
    preds_sorted = sorted(pred_tubes, key=lambda x: x.score, reverse=True)

    gt_matched = [False] * len(gt_tubes)

    for pred in preds_sorted:
        best_iou = 0.0
        best_j   = -1

        for j, gt in enumerate(gt_tubes):
            if gt_matched[j]:
                continue
            if gt.player_idx != pred.player_idx:
                continue
            iou = temporal_iou(
                (pred.start_frame, pred.end_frame),
                (gt.start_frame,   gt.end_frame),
            )
            if iou > best_iou:
                best_iou = iou
                best_j   = j

        matched = (best_iou >= iou_threshold
                   and best_j >= 0
                   and gt_tubes[best_j].class_idx == pred.class_idx)

        if matched:
            gt_matched[best_j] = True

        class_preds[pred.class_idx].append({
            "score":   pred.score,
            "matched": matched,
        })

    return class_preds, n_gt_per_cls


# ──────────────────────────────────────────────────────────────────────────────
# GT tube builder from dataset labels
# ──────────────────────────────────────────────────────────────────────────────

def labels_to_gt_tubes(
    labels: torch.Tensor,   # N×T  int64
) -> list[ActionTube]:
    """
    Convert dense label tensor to GT ActionTube list.
    Background frames (class 0) are skipped.
    """
    from .utils.action_tubes import run_length_encode
    N, T = labels.shape
    tubes = []
    for n in range(N):
        seq   = labels[n].cpu().numpy()
        scores = np.ones(T, dtype=np.float32)  # GT confidence is always 1
        segs  = run_length_encode(seq, scores)
        for (sf, ef, cls_idx, _) in segs:
            if cls_idx == BACKGROUND:
                continue
            tubes.append(ActionTube(
                player_idx  = n,
                start_frame = sf,
                end_frame   = ef,
                class_idx   = cls_idx,
                class_name  = IDX_TO_LABEL.get(cls_idx, "unknown"),
                score       = 1.0,
            ))
    return tubes


# ──────────────────────────────────────────────────────────────────────────────
# Full evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    model:         TAADWithGNN,
    val_loader,
    device:        torch.device,
    iou_thresholds: list[float] = (0.2, 0.5),
    conf_threshold: float = CONF_THRESHOLD,
) -> dict:
    """
    Runs the full mAP evaluation and prints a class-level breakdown.

    Returns:
        results dict: {
            "mAP@<thr>": float,
            "AP_per_class@<thr>": {class_name: float},
            "tp_fp_summary": {...},
        }
    """
    model.eval()

    # Accumulators: one per IoU threshold
    all_class_preds  = {thr: defaultdict(list) for thr in iou_thresholds}
    all_n_gt_per_cls = {thr: defaultdict(int)  for thr in iou_thresholds}

    tp_total = fp_total = 0   # for high-recall mode analysis

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            if batch is None:
                continue

            clip        = batch["clip"].to(device)
            bboxes      = batch["bboxes"].to(device)
            labels      = batch["labels"]              # B×N×T  (stay on CPU for GT)
            player_mask = batch["player_mask"].to(device)

            gs = extract_game_state(
                bboxes, player_mask=player_mask,
                use_proxy=USE_PROXY_POSITIONS,
            )

            out = model(
                clip        = clip,
                bboxes      = bboxes,
                positions   = gs["positions"].to(device),
                velocities  = gs["velocities"].to(device),
                team_ids    = gs["team_ids"].to(device),
                player_mask = gs["player_mask"].to(device),
            )

            B = clip.shape[0]
            for b in range(B):
                # Predicted tubes
                pred_results = build_action_tubes(
                    probs         = out["probs"],
                    team_ids      = gs["team_ids"],
                    fps           = float(FRAME_RATE),
                    conf_threshold= conf_threshold,
                    batch_idx     = b,
                )
                pred_tubes = pred_results["action_tubes"]

                # GT tubes from label tensor
                gt_tubes = labels_to_gt_tubes(labels[b])   # N×T

                for thr in iou_thresholds:
                    cp, ng = match_tubes(pred_tubes, gt_tubes, iou_threshold=thr)
                    for cls_idx, plist in cp.items():
                        all_class_preds[thr][cls_idx].extend(plist)
                    for cls_idx, cnt in ng.items():
                        all_n_gt_per_cls[thr][cls_idx] += cnt

                # TP/FP counts at IoU=0.2, conf=0.5 (Section 4.3.1)
                for p in pred_tubes:
                    if p.score >= conf_threshold:
                        gt_matched = any(
                            temporal_iou(
                                (p.start_frame, p.end_frame),
                                (g.start_frame, g.end_frame),
                            ) >= 0.2 and g.class_idx == p.class_idx
                            for g in gt_tubes
                        )
                        if gt_matched:
                            tp_total += 1
                        else:
                            fp_total += 1

    # ── Compute mAP ──────────────────────────────────────────────────────────
    results = {}
    for thr in iou_thresholds:
        aps = {}
        for cls_idx in range(1, NUM_CLASSES):   # skip background
            preds  = all_class_preds[thr].get(cls_idx, [])
            n_gt   = all_n_gt_per_cls[thr].get(cls_idx, 0)
            ap     = compute_ap_for_class(preds, n_gt)
            aps[IDX_TO_LABEL[cls_idx]] = ap

        valid_aps = [v for v in aps.values() if not np.isnan(v)]
        mean_ap   = float(np.mean(valid_aps)) if valid_aps else 0.0

        key = f"mAP@{thr}"
        results[key] = mean_ap
        results[f"AP_per_class@{thr}"] = aps

        print(f"\n── mAP @ IoU={thr} ─────────────────────────────────")
        print(f"  Overall mAP: {mean_ap * 100:.2f}%")
        for cls_name, ap in sorted(aps.items()):
            tag = f"{ap * 100:6.2f}%" if not np.isnan(ap) else "   N/A"
            print(f"  {cls_name:<18}: {tag}")

    results["tp_fp_summary"] = {
        "true_positives":  tp_total,
        "false_positives": fp_total,
        "precision": tp_total / max(tp_total + fp_total, 1),
    }
    print(f"\n── TP/FP summary (IoU=0.2, conf={conf_threshold}) ─────")
    print(f"  TP={tp_total}  FP={fp_total}  "
          f"Precision={results['tp_fp_summary']['precision']:.3f}")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate TAAD+GNN action detector")
    p.add_argument("--ckpt",           required=True)
    p.add_argument("--data",           required=True)
    p.add_argument("--iou_thresholds", nargs="+", type=float, default=[0.2, 0.5])
    p.add_argument("--conf",           type=float, default=CONF_THRESHOLD)
    p.add_argument("--device",         default="cuda")
    p.add_argument("--workers",        type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = TAADWithGNN(cfg=cfg_module, num_classes=NUM_CLASSES)
    ck    = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck["model"])
    model.to(device)

    _, val_loader = build_dataloaders(
        root_dir=args.data,
        num_workers=args.workers,
    )

    evaluate(
        model          = model,
        val_loader     = val_loader,
        device         = device,
        iou_thresholds = args.iou_thresholds,
        conf_threshold = args.conf,
    )
