# Action Detection Module — TAAD + GNN
### Part 1 of the Football Intelligence Pipeline (Part 5, Section 5.3)

Implements **"Game State and Spatio-temporal Action Detection in Soccer using
Graph Neural Networks and 3D Convolutional Networks"** (Ochin et al., 2025)
as Module 1 of your pipeline.

---

## Architecture

```
clip  B×3×T×H×W
  └── X3D-S backbone (blocks 1-5)
        └── FPN (last 3 blocks → 192ch fused map)
              └── ROI Align along tracklets
                    └── Φ_X3D  B×N×T×192          ← Equation 1

game state (pos, vel, team, Φ_proj)
  └── Build graph G=(V,E):
        • Node per player per timestep
        • Temporal edges (same player, adjacent frames)
        • Spatial edges  (K=6 nearest neighbours)
  └── EdgeConv × 3 layers                          ← Equation 3
        └── h^K  B×N×T×128

concat [ Φ_X3D | h^K ]  →  B×N×T×320
  └── TCN (3 × Conv1D + BN + ReLU + Dropout)
        └── MLP
              └── logits  B×N×T×NUM_CLASSES        ← Section 3.4
```

---

## Your Dataset → Label Mapping

| Folder | Paper class | Index |
|---|---|---|
| `corner` | corner | 3 |
| `foul` | tackle | 9 |
| `freekick` | free-kick | 4 |
| `goal` | shot | 2 |
| `goalkick` | goal-kick | 5 |
| `longpass` | pass | 1 |
| `ontarget` | shot | 2 |
| `penalty` | penalty | 6 |
| `shortpass` | pass | 1 |
| `substitution` | substitution | 7 |
| `throw-in` | throw-in | 8 |

`goal` and `ontarget` share index 2 (`shot`).  
Edit `configs/labels.py` to split them if needed.

---

## Installation

```bash
# 1. PyTorch (match your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 2. Video backbone
pip install pytorchvideo

# 3. Graph Neural Networks
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# 4. Other dependencies
pip install opencv-python numpy tqdm scipy
```

---

## File Structure

```
action_detection/
├── configs/
│   ├── config.py          # All hyperparameters (edit here)
│   └── labels.py          # Label mapping + API-2 event names
├── models/
│   ├── visual_extractor.py  # X3D + FPN + ROI Align → Φ_X3D
│   ├── gnn_game_state.py    # EdgeConv GNN → h^K
│   ├── tcn_head.py          # TCN + MLP → logits
│   └── taad_gnn.py          # Full TAAD+GNN model
├── utils/
│   ├── dataset.py           # ClipDataset + DataLoader
│   ├── game_state.py        # Game-state builder (proxy + real)
│   └── action_tubes.py      # Tube construction + JSON output
├── train.py                 # Training script
├── infer.py                 # Inference + ActionDetectionModule
├── evaluate.py              # mAP evaluation (Section 4.1)
└── requirements.txt
```

---

## Usage

### Train
```bash
python -m action_detection.train \
    --data  data/raw_videos/clip_dataset \
    --ckpt  checkpoints/taad_gnn.pt \
    --device cuda
```

### Evaluate
```bash
python -m action_detection.evaluate \
    --ckpt  checkpoints/taad_gnn.pt \
    --data  data/raw_videos/clip_dataset \
    --iou_thresholds 0.2 0.5 \
    --conf 0.5
```

### Infer on a single clip
```bash
python -m action_detection.infer \
    --ckpt   checkpoints/taad_gnn.pt \
    --video  data/raw_videos/clip_dataset/corner/clip_001.avi \
    --output results/events.json
```

### Use in your pipeline (Part 5)
```python
from action_detection import ActionDetectionModule

# Initialise once
detector = ActionDetectionModule("checkpoints/taad_gnn.pt", device="cuda")

# Process a clip
output = detector.process("path/to/clip.avi")

# Feed to Formation Module (API-2 format, Part 5 §5.10)
event_json   = output["event_json"]    # str — JSON for Module 2
event_dicts  = output["event_dicts"]   # list[EventDict]
action_tubes = output["action_tubes"]  # list[ActionTube]
frame_probs  = output["frame_probs"]   # np.ndarray  N×T×C
```

#### API-2 event JSON example
```json
[
  {
    "start_time":  2.040,
    "end_time":    2.160,
    "start_frame": 51,
    "end_frame":   54,
    "team":        0,
    "player":      0,
    "event_type":  "corner",
    "confidence":  0.8731
  }
]
```

---

## Three Output Formats

| Format | Variable | Use in pipeline |
|---|---|---|
| **JSON event dict** | `output["event_json"]` | API-2 → Formation Module (§5.10) |
| **Action tubes** | `output["action_tubes"]` | Tube-level stats, evaluation |
| **Per-frame probs** | `output["frame_probs"]` | N×T×C numpy — Offensive Prediction |

---

## GNN Game-State Proxy

Until **homography + 2-D pitch reconstruction** is ready:

- **Position** = normalised bbox centre (cx/W, cy/H) — screen space
- **Velocity** = finite-difference of position over adjacent frames
- **Team ID** = 0 for all players (unknown)

To switch to real pitch coordinates, open `utils/game_state.py` and replace
the body of `extract_game_state()`.  No other file needs changing.

The GNN graph and EdgeConv layers are **fully wired** — only the input
feature values change when real coords are plugged in.

---

## Key Hyperparameters (`configs/config.py`)

| Parameter | Value | Paper reference |
|---|---|---|
| `CLIP_FRAMES` | 50 | Section 3.2.3 |
| `FRAME_RATE` | 25 FPS | Section 3.2.3 |
| `IMG_H × IMG_W` | 352×640 | Section 3.2.3 |
| `X3D_MODEL` | x3d_s | Section 3.2.2 |
| `X3D_FEAT_DIM` | 192 (D) | Section 3.2.2 |
| `VISUAL_PROJ_DIM` | 64 (D') | Equation 2 |
| `K_NEIGHBORS` | 6 | Section 3.3.2 |
| `GNN_LAYERS` | 3 | Section 3.3.3 |
| `LR` | 5e-4 | Section 3.6 |
| `GRAD_ACCUM_STEPS` | 20 | Section 3.6 |
| `NUM_EPOCHS` | 13 | Section 3.6 |
| `LR_DROP_EPOCH` | 10 | Section 3.6 |
| `CONF_THRESHOLD` | 0.5 | Section 4.3.1 |
| `IOU_THRESHOLD` | 0.2 | Section 4.1 |

---

## Upgrading to Real Tracker

When ByteTrack (or any tracker) is integrated:

1. Replace `utils/dataset.py::_dummy_bboxes()` with real tracker bboxes.
2. Replace `utils/game_state.py::extract_game_state()` with real pitch coords.
3. Set `configs/config.py::USE_PROXY_POSITIONS = False`.
4. Populate `team_ids` (0/1) from re-ID or jersey detection.

The model architecture and training loop require **zero changes**.
