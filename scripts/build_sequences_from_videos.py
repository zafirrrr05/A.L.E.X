import os
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from src.perception.detector import PlayerBallDetector
from src.perception.tracker import SimpleTracker
from src.preprocessing.team_assigner import TeamAssigner
from src.preprocessing.sequence_builder import SequenceBuilder
from src.preprocessing.save_sequences import save_sequences, get_next_sequence_index
from src.utils.visualization import draw_detections, draw_tracks, BallInterpolator, BallDetectionMemory

# python -m scripts.build_sequences_from_videos
VIDEOS_DIR = "data/raw_videos/video_dataset/DFL Bundesliga Data Shootout/simple_sample_train"
SEQUENCE_DIR = "data/sequences"
DONE_DIR = os.path.join(SEQUENCE_DIR, "_done")
DETECTION_DIR = "data/detections"
TRACK_DIR = "data/tracks"


# ---------------------------------------------------------
# detection cache helpers
# ---------------------------------------------------------

def save_detections_npz(path, detections):
    np.savez_compressed(path, detections=detections)


def load_detections_npz(path):
    return np.load(path, allow_pickle=True)["detections"].tolist()


# ---------------------------------------------------------
# sequence cache helpers
# ---------------------------------------------------------
def is_video_done(video_path):
    name = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.exists(os.path.join(DONE_DIR, name + ".done"))


def mark_video_done(video_path, start_index, num_sequences):
    name = os.path.splitext(os.path.basename(video_path))[0]
    path = os.path.join(DONE_DIR, name + ".done")

    with open(path, "w") as f:
        f.write(f"start_index={start_index}\n")
        f.write(f"num_sequences={num_sequences}\n")


# ---------------------------------------------------------

def process_video(video_path, global_start_index, detector):

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    det_out_dir = os.path.join(DETECTION_DIR, video_name)
    os.makedirs(det_out_dir, exist_ok=True)

    tracker = SimpleTracker()
    team_assigner = TeamAssigner()
    ball_interpolator = BallInterpolator(max_gap=30)
    ball_memory = BallDetectionMemory(max_gap=10)

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    track_video_path = os.path.join(TRACK_DIR, f"{video_name}.mp4")
    writer = cv2.VideoWriter(
        track_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H)
    )

    track_history = defaultdict(list)

    fitted = False
    frame_id = 0

    for _ in tqdm(range(total), desc=video_name):

        ret, frame = cap.read()
        if not ret:
            break

        det_npz_path = os.path.join(
            det_out_dir, f"frame_{frame_id:06d}.npz"
        )

        # -------------------------------------------------
        # DETECTION (cached)
        # -------------------------------------------------
        if os.path.exists(det_npz_path):
            detections = load_detections_npz(det_npz_path)
        else:
            detections = detector.detect(frame)
            save_detections_npz(det_npz_path, detections)

        # -------------------------------------------------
        # TRACKING
        # -------------------------------------------------
        tracks = tracker.update(detections)

        # -------------------------------------------------
        # TEAM ASSIGNMENT
        # -------------------------------------------------
        if not fitted:
            fitted = team_assigner.fit(frame, tracks)

        if fitted:
            tracks = team_assigner.assign(frame, tracks)

        # ---------- SAVE DETECTION JPG ----------
        # keep your original behaviour
        if frame_id % 10 == 0:
            ball_bbox = ball_memory.update(frame_id, detections)
            det_img = draw_detections(frame, detections, ball_bbox=ball_bbox)
            det_path = os.path.join(det_out_dir, f"frame_{frame_id:06d}.jpg")
            if not os.path.exists(det_path):
                cv2.imwrite(det_path, det_img)

        # ---------- SAVE TRACKED VIDEO ----------
        ball_bbox = ball_interpolator.update_from_detections(frame_id, detections)
        vis = draw_tracks(frame, tracks, ball_bbox=ball_bbox)
        writer.write(vis)

        track_history[frame_id] = tracks
        frame_id += 1

    cap.release()
    writer.release()

    print(f"[INFO] {video_name} frames processed: {frame_id}")

    builder = SequenceBuilder(window=50, stride=10)
    sequences = builder.build(track_history)

    print(f"[INFO] sequences generated: {len(sequences)}")

    save_sequences(sequences, SEQUENCE_DIR, start_index=global_start_index)
    mark_video_done(
        video_path,
        global_start_index,
        len(sequences)
    )

    return global_start_index + len(sequences)


def main():

    os.makedirs(SEQUENCE_DIR, exist_ok=True)
    os.makedirs(DONE_DIR, exist_ok=True)
    os.makedirs(DETECTION_DIR, exist_ok=True)
    os.makedirs(TRACK_DIR, exist_ok=True)

    videos = sorted([
        os.path.join(VIDEOS_DIR, f)
        for f in os.listdir(VIDEOS_DIR)
        if f.lower().endswith(".mp4")
    ])

    detector = PlayerBallDetector()

    global_index = get_next_sequence_index(SEQUENCE_DIR)

    print("[INFO] starting sequence index:", global_index)
    for v in videos:
        if is_video_done(v):
            print("[SKIP] already processed:", v)
            continue

        print("\nProcessing:", v)
        global_index = process_video(v, global_index, detector)

    print("\nAll done. Total sequences:", global_index)


if __name__ == "__main__":
    main()