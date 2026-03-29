import numpy as np
import supervision as sv


class Track:
    def __init__(self, track_id, bbox, class_id):
        self.id = track_id
        self.class_id = class_id

        # important for team assignment later, so we can carry it across frames
        self.team_id = None

        self.age = 0
        self.time_since_update = 0

        x1, y1, x2, y2 = bbox
        self.bbox = bbox
        self.cx = (x1 + x2) / 2.0
        self.cy = (y1 + y2) / 2.0
        self.vx = 0.0
        self.vy = 0.0

    def update(self, bbox, prev_cx=None, prev_cy=None):
        x1, y1, x2, y2 = bbox
        new_cx = (x1 + x2) / 2.0
        new_cy = (y1 + y2) / 2.0

        if prev_cx is not None:
            self.vx = new_cx - prev_cx
            self.vy = new_cy - prev_cy

        self.cx = new_cx
        self.cy = new_cy
        self.bbox = bbox
        self.time_since_update = 0
        self.age += 1


class SimpleTracker:

    def __init__(
        self,
        track_activation_threshold=0.25,
        lost_track_buffer=60,       # frames — 60 = ~2s at 30fps
        minimum_matching_threshold=0.8,
        frame_rate=30,
    ):
        self.byte_tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )

        # track_id -> Track, so we can carry state across frames
        self._track_store: dict[int, Track] = {}

    def update(self, detections: list[dict]) -> list[Track]:
        """
        detections: list of dicts with keys:
            "bbox"       -> [x1, y1, x2, y2]
            "confidence" -> float
            "class"      -> int
        """
        if not detections:
            return []

        xyxy       = np.array([d["bbox"]       for d in detections], dtype=np.float32)
        confidence = np.array([d["confidence"] for d in detections], dtype=np.float32)
        class_id   = np.array([d["class"]      for d in detections], dtype=int)

        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )

        sv_detections = self.byte_tracker.update_with_detections(sv_detections)

        tracks = []
        for i in range(len(sv_detections)):
            track_id = int(sv_detections.tracker_id[i])
            bbox     = sv_detections.xyxy[i].tolist()
            cls      = int(sv_detections.class_id[i])

            if track_id in self._track_store:
                existing = self._track_store[track_id]
                existing.update(bbox, prev_cx=existing.cx, prev_cy=existing.cy)
                tracks.append(existing)
            else:
                tr = Track(track_id, bbox, cls)
                self._track_store[track_id] = tr
                tracks.append(tr)

        # clean up tracks that bytetrack dropped
        active_ids = {int(sv_detections.tracker_id[i]) for i in range(len(sv_detections))}
        self._track_store = {k: v for k, v in self._track_store.items() if k in active_ids}

        return tracks