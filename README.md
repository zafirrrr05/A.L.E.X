# ⚽ ALEX — Advanced Learning Engine for eXplainable football tactics

> An end-to-end football tactical AI system that transforms raw match video into structured tactical intelligence — using computer vision, graph neural networks, and self-supervised learning.

---

## 🧠 What ALEX Does

ALEX ingests raw match footage and produces structured tactical analysis at the clip level. It detects and tracks every player and the ball, builds relational graph representations of each frame, and passes those graphs through a hierarchical deep learning model that simultaneously predicts:

- **Tactical formation** (e.g. 4-4-2, 4-3-3, 3-5-2, 4-2-3-1)
- **Set piece type** (corner, free-kick, throw-in, open play)
- **Pass network** (most likely next passer among 22 players)
- **Player movement** (predicted Δx, Δy for all 22 outfield players)
- **Pass quality** (short pass vs. long pass classification)

The system is designed to scale to full-match analysis and currently supports video clips via an automated sequence-building pipeline.

---

## 🏗️ Architecture Overview

```
Raw Video
    │
    ▼
┌─────────────────────────────┐
│  Perception Layer           │  YOLOv8x  →  ByteTrack
│  detector.py / tracker.py   │  Detects: player, goalkeeper, ball, referee
└──────────────┬──────────────┘
               │
    ┌──────────▼──────────┐
    │  Preprocessing       │  Team assignment via jersey colour (K-Means)
    │  team_assigner.py    │  Sequence building from tracked clips
    │  sequence_builder.py │  Saves .npz files per clip (50 frames each)
    └──────────┬───────────┘
               │
    ┌──────────▼───────────────────────────────┐
    │  Hierarchical Dual GATv2 Model           │
    │  dual_gatv2_model.py                     │
    │                                          │
    │  Graph 1 (Player-level)                  │
    │    23 nodes × 10 features                │
    │    4 edge features                       │
    │    2-layer GATv2  →  pool per team       │
    │                    ↓                     │
    │  Graph 2 (Team-level)                    │
    │    3 nodes (Team A, Team B, Ball)        │
    │    Injected with G1 pooled embeddings    │
    │                    ↓                     │
    │  Temporal: Bi-LSTM + Soft Attention      │
    │    50 frames  →  clip embedding (256-d)  │
    │                    ↓                     │
    │  Task Heads (5 simultaneous outputs)     │
    └──────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
sport_tactical_ai/
│
├── src/
│   ├── perception/
│   │   ├── detector.py          # YOLOv8 wrapper — detects players, ball, referee
│   │   └── tracker.py           # ByteTrack multi-object tracker
│   │
│   ├── preprocessing/
│   │   ├── team_assigner.py     # Jersey colour extraction + K-Means team assignment
│   │   ├── sequence_builder.py  # Builds 50-frame .npz sequences from tracked clips
│   │   ├── jersey_color_extractor.py
│   │   └── save_sequences.py
│   │
│   ├── models/
│   │   ├── dual_gatv2_model.py  # Full hierarchical GATv2 + Bi-LSTM + 5 task heads
│   │   └── team_tactical_net.py # Backbone for SSL pre-training
│   │
│   ├── training/
│   │   ├── graph_dataset.py       # .npz → PyG Data objects
│   │   ├── ssl_dataset.py         # Self-supervised pretext task datasets
│   │   ├── ssl_trainer.py         # 5-task SSL trainer (future, masked, possession, order, contrastive)
│   │   ├── pass_dataset.py        # Pass quality dataset
│   │   ├── formation_analysis.py  # Formation label extraction
│   │   └── space_targets.py       # Space control targets
│   │
│   ├── inference/                 # (In development)
│   └── homograph/                 # Pitch homography estimation
│
├── scripts/
│   ├── build_sequences_from_videos.py  # Converts AVI clips → labelled .npz sequences
│   ├── build_sequences_from_images.py  # Converts image folders → sequences
│   ├── build_homograph.py              # Fits pitch keypoint homography
│   ├── train_pass_gatv2.py             # Trains GATv2 pass quality head
│   ├── train_pass_quality.py           # Trains pass quality classifier
│   ├── train_formation.py              # Trains formation classification head
│   ├── train_space_control.py          # Trains space control prediction
│   ├── train_ssl.py                    # Runs self-supervised pre-training
│   └── sanity_check_balanced.py        # Dataset balance diagnostics
│
├── data/
│   ├── raw_videos/       # Source AVI match clips (by event class)
│   ├── raw_images/       # Source JPEG frames
│   ├── detections/       # YOLO detection outputs (JSON)
│   ├── tracks/           # ByteTrack trajectory outputs
│   ├── sequences/        # Built .npz clip sequences (training-ready)
│   └── pitch_coords/     # Pitch keypoint homography data
│
├── checkpoints/          # Trained model weights
│   ├── pass_gatv2.pt
│   ├── pass_quality.pt
│   └── ssl_encoder.pt
│
├── training_yolo/        # YOLOv8 fine-tuning setup (Roboflow dataset)
├── config.yaml           # YOLO class config (ball, goalkeeper, player, referee)
└── requirements.txt
```

---

## 🔬 The Model: Hierarchical Dual GATv2

The core model (`HierarchicalDualGATv2`) is a multi-scale graph transformer architecture built for football tactical understanding.

### Graph 1 — Player-Level GATv2
- **23 nodes**: 11 players per team + 1 ball node
- **10 node features** per player: position, velocity, team identity, role
- **4 edge features**: distance, angle, relative velocity, team membership
- Two GATv2 convolutional layers with multi-head attention (4 heads → 1 head)
- Output: per-node embeddings + graph-level mean/max pooled representation

### Graph 2 — Team-Level GATv2
- **3 nodes**: Team A, Team B, Ball (each initialised from G1 pooled embeddings)
- Captures inter-team relational dynamics at a coarser tactical scale
- Output: team-level graph embedding (64-d)

### Temporal Reasoning — Bi-LSTM + Soft Attention
- Processes 50 consecutive frames as a sequence
- Bidirectional LSTM (2 layers, 128 hidden) with learnable soft attention pooling
- Produces a 256-d clip-level embedding encoding temporal tactical patterns

### Task Heads (Multi-Task Learning)
| Head | Task | Output |
|---|---|---|
| `formation` | Formation classification | 5 classes |
| `set_piece` | Set piece type | 4 classes |
| `pass_net` | Next passer prediction | 22 logits |
| `movement` | Player movement prediction | 44 values (22 × Δx, Δy) |
| `pass_quality` | Pass quality (short/long) | 2 classes |

---

## 🤖 Self-Supervised Pre-Training

Before supervised fine-tuning, the backbone (`TacticalModel`) is pre-trained on 5 self-supervised pretext tasks — removing the dependency on expensive human-labelled data:

| Task | What the model learns |
|---|---|
| **Future state prediction** | Predicts ball position at the next timestep |
| **Masked player modelling** | Reconstructs masked player positions (inspired by BERT) |
| **Possession continuity** | Classifies which team holds possession in a clip |
| **Temporal order** | Determines which of two clips comes first |
| **Contrastive learning** | Pulls augmented views of the same clip together (NT-Xent) |

---

## 🚀 Roadmap

The system is actively being developed. The next priorities are:

1. **Frame-level event classification** — Fine-tune EfficientNet-B0 on 58,000+ labelled frames across 10 event classes (pass, shot, foul, corner, etc.)
2. **Real-label training pipeline** — Replace geometric heuristics with ground-truth labels extracted from 3,800+ AVI event clips (expected accuracy jump: 0.53 → 0.70+)
3. **GATv2 head retraining** — Retrain pass quality, set piece, and new event heads with real labels
4. **Visual + graph fusion** — Fuse EfficientNet frame features with GATv2 clip embeddings for richer representations
5. **Formation improvements** — Centroid-relative normalisation, multi-scale temporal windows, Dynamic Role Assignment Module

---

## 🛠️ Installation

```bash
git clone https://github.com/your-username/sport_tactical_ai.git
cd sport_tactical_ai

python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS / Linux

pip install -r requirements.txt
```

**Requirements**: Python 3.10+, CUDA 11.8+ (recommended), PyTorch 2.1+, torch-geometric

---

## ⚡ Quick Start

### 1. Build sequences from raw video clips
```bash
python scripts/build_sequences_from_videos.py
```

### 2. Run self-supervised pre-training
```bash
python scripts/train_ssl.py
```

### 3. Train the pass quality head
```bash
python scripts/train_pass_gatv2.py
```

### 4. Train formation classification
```bash
python scripts/train_formation.py
```

### 5. Sanity check dataset balance
```bash
python scripts/sanity_check_balanced.py
```

---

## 📦 Data

The system uses two data sources:

| Source | Description |
|---|---|
| **Roboflow Football Dataset** | 58,000+ labelled frames for YOLO fine-tuning and frame-level classification ([CC BY 4.0](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc)) |
| **Action video clips** | 3,800+ AVI clips organised by event class (shortpass, longpass, goal, foul, corner, etc.) |

Raw data is **not tracked by Git** — see `.gitignore`.

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Object detection | YOLOv8x (Ultralytics) |
| Multi-object tracking | ByteTrack |
| Graph neural networks | PyTorch Geometric — GATv2Conv |
| Temporal modelling | PyTorch Bi-LSTM |
| Self-supervised learning | Custom SSL trainer (5 pretext tasks) |
| Computer vision | OpenCV |
| Data processing | NumPy, SciPy, scikit-learn |
| Tracking utilities | FilterPy, LAP |

---

## 📄 License

This project is for research and educational purposes.
