import numpy as np
import cv2
from sklearn.cluster import KMeans


class TeamClassifier:
    """
    Stable team assignment:
    - First keyframe initializes KMeans
    - Afterwards assignments are done via nearest centroid
    - Persist track_id -> team mapping to eliminate flicker
    """

    def __init__(self):
        self.kmeans = None
        self.centroids = None
        self.initialized = False

        # persistent dictionary: track_id -> team ("A" or "B")
        self.team_memory = {}

    def _extract_jersey_color(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)

        H, W = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        h, w = crop.shape[:2]
        jersey = crop[int(h * 0.25):int(h * 0.75),
                      int(w * 0.25):int(w * 0.75)]

        if jersey.size == 0:
            return None

        return np.mean(jersey.reshape(-1, 3), axis=0)

    def initialize_teams(self, frame, tracks):
        """
        Called once: at first keyframe.
        Assigns stable team color for the rest of the clip.
        """
        colors = []
        track_ids = []

        for t in tracks:
            if t["class"] != 2:  # players only
                continue
            col = self._extract_jersey_color(frame, t["bbox"])
            if col is not None:
                colors.append(col)
                track_ids.append(t["track_id"])

        if len(colors) < 2:
            print("[TeamClassifier] Not enough player colors; skipping.")
            return tracks

        colors = np.array(colors)

        self.kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        labels = self.kmeans.fit_predict(colors)
        self.centroids = self.kmeans.cluster_centers_
        self.initialized = True

        # assign stable team for each track_id
        for tid, lab in zip(track_ids, labels):
            team = "A" if lab == 0 else "B"
            self.team_memory[tid] = team

        # fill into tracks
        for t in tracks:
            if t["class"] != 2:
                continue
            tid = t["track_id"]
            t["team"] = self.team_memory.get(tid, "unknown")

        return tracks

    def assign_teams(self, frame, tracks):
        """
        After initialization, assign teams using stored stable mapping.
        """
        if not self.initialized:
            return tracks

        for t in tracks:
            if t["class"] != 2:
                continue

            tid = t["track_id"]

            if tid in self.team_memory:
                t["team"] = self.team_memory[tid]
            else:
                # assign new track id by nearest centroid
                col = self._extract_jersey_color(frame, t["bbox"])
                if col is None:
                    t["team"] = "unknown"
                    continue

                d0 = np.linalg.norm(col - self.centroids[0])
                d1 = np.linalg.norm(col - self.centroids[1])
                team = "A" if d0 < d1 else "B"

                self.team_memory[tid] = team
                t["team"] = team

        return tracks