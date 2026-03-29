# ======================================================================
# build_homograph.py — FINAL VERSION (Broadcast Camera Corrected)
# ======================================================================

import os
import glob
import json
import csv
import cv2
import numpy as np
from tqdm import tqdm

# --- HOMOGRAPHY COMPONENTS ---
from src.homograph.track_builder import TrackBuilder
from src.homograph.team_classifier import TeamClassifier
from src.homograph.pitch_detector import PitchDetector
from src.homograph.pitch_model import FIFAPitchModel
from src.homograph.pitch_keypoint_mapper import PitchKeypointMapper
from src.homograph.homography_estimator import HomographyEstimator
from src.homograph.projection import Projector
from src.homograph.visualizer import TacticalVisualizer

# =====================================================================
# CONFIG
# =====================================================================

DETECTIONS_ROOT = r"D:\sport_tactical_ai\data\detections"
OUTPUT_ROOT     = r"D:\sport_tactical_ai\data\pitch_coords"

# 👉 Change this to your clip path if needed
INPUT_CLIP_DIR  = r"D:\sport_tactical_ai\data\detections\A1606b0e6_0 (003)"

clip_name = os.path.basename(INPUT_CLIP_DIR.rstrip("/\\"))
OUT_DIR   = os.path.join(OUTPUT_ROOT, clip_name)
os.makedirs(OUT_DIR, exist_ok=True)

print(f"[run_pipeline] Clip:   {clip_name}")
print(f"[run_pipeline] Input:  {INPUT_CLIP_DIR}")
print(f"[run_pipeline] Output: {OUT_DIR}")

FPS     = 25
PITCH_W = 105
PITCH_H = 68

# =====================================================================
# 1) TRACK BUILDING
# =====================================================================

tb = TrackBuilder(INPUT_CLIP_DIR)
tracks_per_frame = tb.build()
total_frames = len(tracks_per_frame)

print(f"[run_pipeline] Frames: {total_frames}")

# =====================================================================
# 2) LOAD KEYFRAMES (every 10th frame JPG)
# =====================================================================

jpgs = sorted(glob.glob(os.path.join(INPUT_CLIP_DIR, "frame_*.jpg")))

keyframes = {}
for fpath in jpgs:
    fname = os.path.basename(fpath)
    frame_id = int(fname.split("_")[1].split(".")[0])
    keyframes[frame_id] = cv2.imread(fpath)

print(f"[run_pipeline] Loaded {len(keyframes)} keyframes")

# =====================================================================
# 3) DETECT FIELD KEYPOINTS ON ALL KEYFRAMES
# =====================================================================

pd = PitchDetector()
detected_kpts = {}

print("\n[run_pipeline] Detecting pitch keypoints...")
for fid in tqdm(sorted(keyframes.keys()), desc="Pitch Detection"):
    detected_kpts[fid] = pd.detect(keyframes[fid])

# =====================================================================
# 4) FIT HOMOGRAPHY (NO INTERPOLATION)
# =====================================================================

model   = FIFAPitchModel()
mapper  = PitchKeypointMapper()
H_est   = HomographyEstimator()

# Use best keyframe (first with valid detection)
valid_H = None

for fid in sorted(detected_kpts.keys()):
    H = H_est.get_H_from_detected(model, mapper, detected_kpts[fid])
    if H is not None:
        valid_H = H
        best_fid = fid
        break

if valid_H is None:
    raise RuntimeError("NO VALID HOMOGRAPHY FOUND — pitch keypoints too weak!")

print(f"[run_pipeline] Using homography from keyframe: {best_fid}")

# =====================================================================
# 5) TEAM CLASSIFICATION
# =====================================================================

tc = TeamClassifier()
first_kf = sorted(keyframes.keys())[0]
tc.initialize_teams(keyframes[first_kf], tracks_per_frame[first_kf])

for f in range(total_frames):
    # if no frame image cached, use nearest keyframe for team assignment
    nearest = min(keyframes.keys(), key=lambda k: abs(k - f))
    img = keyframes[nearest]
    tracks_per_frame[f] = tc.assign_teams(img, tracks_per_frame[f])

# =====================================================================
# 6) PROJECT ALL TRACKS TO 105x68 PITCH COORDINATES
# =====================================================================

proj = Projector(valid_H, PITCH_W, PITCH_H)
projected = {}

print("\n[run_pipeline] Projecting positions...")
for f in tqdm(range(total_frames), desc="Projecting"):
    projected[f] = proj.project_frame(f, tracks_per_frame[f])

# =====================================================================
# 7) SAVE JSON + CSV OUTPUTS
# =====================================================================

json_path = os.path.join(OUT_DIR, "projected_positions.json")
csv_path  = os.path.join(OUT_DIR, "projected_positions.csv")

with open(json_path, "w") as jf:
    json.dump(projected, jf, indent=2)

with open(csv_path, "w", newline="") as cf:
    writer = csv.writer(cf)
    writer.writerow(["frame", "track_id", "team", "class", "field_x", "field_y"])

    for f, objs in projected.items():
        for o in objs:
            writer.writerow([
                f, o["track_id"], o["team"], o["class"],
                o["field_x"], o["field_y"]
            ])

print(f"[run_pipeline] Saved outputs → {OUT_DIR}")

# =====================================================================
# 8) RENDER TOP-DOWN VIDEO (Top View Tactical Camera)
# =====================================================================

viz = TacticalVisualizer(PITCH_W, PITCH_H, out_w=900, out_h=585)
mp4_path = os.path.join(OUT_DIR, "pitch_view.mp4")

viz.render_video(projected, mp4_path, fps=FPS)

print("\n==============================")
print("   HOMOGRAPHY PIPELINE DONE  ")
print("==============================")