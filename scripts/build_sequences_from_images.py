import os
import cv2
from collections import defaultdict

from src.perception.detector import PlayerBallDetector
from src.perception.tracker import SimpleTracker
from src.preprocessing.team_assigner import TeamAssigner
from src.preprocessing.sequence_builder import SequenceBuilder
from src.preprocessing.save_sequences import save_sequences

from tqdm import tqdm


IMAGES_DIR = "data/raw_videos/dataset_T000/train"
OUT_DIR = "data/sequences"


def process_image_folder(folder_path, global_start_index, detector):

    tracker = SimpleTracker()
    team_assigner = TeamAssigner()

    track_history = defaultdict(list)

    # ONLY .jpg (not .jpeg)
    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(".jpg")
    ])

    frame_id = 0
    fitted = False

    for name in tqdm(image_files, desc=os.path.basename(folder_path)):

        img_path = os.path.join(folder_path, name)
        frame = cv2.imread(img_path)

        if frame is None:
            continue

        detections = detector.detect(frame)
        tracks = tracker.update(detections)

        if not fitted and len(tracks) > 0:
            team_assigner.fit(frame, tracks)
            fitted = True

        if fitted:
            tracks = team_assigner.assign(frame, tracks)

        track_history[frame_id] = tracks
        frame_id += 1

    print(f"[INFO] frames:", frame_id)

    builder = SequenceBuilder(window=50, stride=10)
    sequences = builder.build(track_history)

    print(f"[INFO] sequences generated:", len(sequences))

    save_sequences(sequences, OUT_DIR, start_index=global_start_index)

    return global_start_index + len(sequences)


def main():

    os.makedirs(OUT_DIR, exist_ok=True)

    detector = PlayerBallDetector()

    global_index = 0

    # all jpg are in ONE folder
    global_index = process_image_folder(
        IMAGES_DIR,
        global_index,
        detector
    )

    print("\nAll done. Total sequences:", global_index)


if __name__ == "__main__":
    main()