import os
import glob
import numpy as np
from tqdm import tqdm

# import your real SimpleTracker
from src.perception.tracker import SimpleTracker


class TrackBuilder:
    """
    Loads .npz detections and produces consistent tracks using SimpleTracker.
    Output is a dict of frame_id -> list of objects with bbox, cx, cy, class, track_id.
    """

    def __init__(self, npz_dir):
        self.npz_dir = npz_dir
        self.frame_files = sorted(glob.glob(os.path.join(npz_dir, "frame_*.npz")))
        self.total_frames = len(self.frame_files)

        print(f"[TrackBuilder] Found {self.total_frames} npz frames")

        # SimpleTracker wrapper around ByteTrack
        self.tracker = SimpleTracker()

    def _load_detections(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        return data["detections"].tolist()

    def build(self):
        tracks_per_frame = {}

        print("\n[TrackBuilder] Tracking across frames...")
        for frame_id in tqdm(range(self.total_frames), desc="Tracking"):
            detections = self._load_detections(self.frame_files[frame_id])
            track_objs = self.tracker.update(detections)

            frame_tracks = []
            for tr in track_objs:
                x1, y1, x2, y2 = tr.bbox

                frame_tracks.append({
                    "track_id": tr.id,
                    "class": tr.class_id,
                    "team": tr.team_id if tr.team_id is not None else "unknown",
                    "bbox": tr.bbox,
                    "cx": tr.cx,
                    "cy": tr.cy,
                })

            tracks_per_frame[frame_id] = frame_tracks

        print("[TrackBuilder] Completed track building.")
        return tracks_per_frame