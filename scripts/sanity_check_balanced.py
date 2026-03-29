"""
scripts/sanity_check_balanced.py
─────────────────────────────────
Samples sequences evenly across the full dataset
to get a balanced ground truth for the spot-check.

Run: python -m scripts.sanity_check_balanced
"""

import torch
import glob
import numpy as np
import random

from src.models.dual_gatv2_model import HierarchicalDualGATv2
from src.training.graph_dataset import frame_to_player_graph, frame_to_team_graph
from src.training.pass_utils import pass_quality_label
from src.utils.data_utils import is_valid_sequence
from src.training.pass_utils import detect_pass_events

PITCH_W = 1280.0
PITCH_H = 768.0
random.seed(42)


def normalise(t1, t2, ball):
    t1   = t1.copy().astype(np.float32)
    t2   = t2.copy().astype(np.float32)
    ball = ball.copy().astype(np.float32)
    t1[:, :, 0] /= PITCH_W;  t1[:, :, 1] /= PITCH_H
    t2[:, :, 0] /= PITCH_W;  t2[:, :, 1] /= PITCH_H
    ball[:, 0]  /= PITCH_W;  ball[:, 1]  /= PITCH_H
    return t1, t2, ball


def predict(model, seq):
    with torch.no_grad():
        out      = model([seq])
        logits = out["pass_quality"]
        probs = torch.softmax(logits, dim=1)
        pred  = logits.argmax(dim=1).item()
    return pred, probs[0]


# ── load model ──
model = HierarchicalDualGATv2()
model.load_state_dict(
    torch.load("checkpoints/pass_gatv2.pt", map_location="cpu")
)
model.eval()
print("Model loaded.")

# ── collect balanced samples from across full dataset ──
all_files = sorted(glob.glob("data/sequences/*.npz"))
random.shuffle(all_files)

good_samples = []   # (seq, label=1)
bad_samples  = []   # (seq, label=0)

print("Collecting balanced samples (target: 50 good + 50 bad)...")

for fpath in all_files:
    if len(good_samples) >= 50 and len(bad_samples) >= 50:
        break

    if not is_valid_sequence(fpath):
        continue

    d  = np.load(fpath)
    t1_raw   = d["players_team1"].astype(np.float32)
    t2_raw   = d["players_team2"].astype(np.float32)
    ball_raw = d["ball"].astype(np.float32)

    t1, t2, ball = normalise(t1_raw, t2_raw, ball_raw)

    t1_t   = torch.from_numpy(t1)
    t2_t   = torch.from_numpy(t2)
    ball_t = torch.from_numpy(ball)

    # only evaluate at actual detected pass events
    events = detect_pass_events(t1_t, t2_t, ball_t)

    for t_pass in events:
        start = t_pass - 5
        end   = t_pass + 10
        if start < 0 or end > t1.shape[0]:
            continue

        y = pass_quality_label(t1_t, t2_t, ball_t, t_pass=t_pass)
        if y is None:
            continue

        seq = []
        for i in range(start, end):
            pg = frame_to_player_graph(t1[i], t2[i], ball[i])
            tg = frame_to_team_graph(t1[i],  t2[i], ball[i])
            seq.append((pg, tg))

        if y == 1 and len(good_samples) < 50:
            good_samples.append((seq, y))
        elif y == 0 and len(bad_samples) < 50:
            bad_samples.append((seq, y))

        if len(good_samples) >= 50 and len(bad_samples) >= 50:
            break

print(f"Collected: good={len(good_samples)}  bad={len(bad_samples)}")

# ── evaluate ──
all_samples = good_samples + bad_samples
random.shuffle(all_samples)

correct           = 0
total             = len(all_samples)
good_pred_on_good = 0
bad_pred_on_bad   = 0
n_good            = len(good_samples)
n_bad             = len(bad_samples)

for seq, y in all_samples:
    pred, probs = predict(model, seq)
    correct           += int(pred == y)
    good_pred_on_good += int(pred == 1 and y == 1)
    bad_pred_on_bad   += int(pred == 0 and y == 0)

acc         = correct / max(total, 1)
recall_good = good_pred_on_good / max(n_good, 1)
recall_bad  = bad_pred_on_bad   / max(n_bad,  1)
random_base = 0.5   # balanced dataset → random = 50%

print()
print("=" * 55)
print("BALANCED SPOT-CHECK RESULTS")
print("=" * 55)
print(f"Total samples   : {total}  (good={n_good}  bad={n_bad})")
print(f"Accuracy        : {acc:.3f}")
print(f"Random baseline : 0.500  (balanced dataset)")
print(f"Recall good     : {recall_good:.3f}  ({good_pred_on_good}/{n_good})")
print(f"Recall bad      : {recall_bad:.3f}  ({bad_pred_on_bad}/{n_bad})")
print()

if acc >= 0.65 and recall_good > 0.50 and recall_bad > 0.50:
    verdict = "GOOD — safe to move forward with overnight jobs"
elif acc >= 0.58 and (recall_good > 0.40 or recall_bad > 0.40):
    verdict = "OK — model learned something, proceed but retrain later"
else:
    verdict = "RETRAIN — model is biased, fix before overnight jobs"

print(f"Verdict: {verdict}")
print("=" * 55)