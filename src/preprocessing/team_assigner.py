import numpy as np
from sklearn.cluster import KMeans
from .jersey_color_extractor import extract_jersey_color


class TeamAssigner:
    def __init__(self, n_teams=2, min_samples=40):
        self.n_teams = n_teams
        self.min_samples = min_samples
        self.kmeans = None
        self.team_colors = None

        # 🔴 persistent across frames
        self._color_buffer = []

    def is_player(self, track):
        return getattr(track, "class_id", -1) == 2

    # ----------------------------
    # FIT — called ONCE (or few times)
    # ----------------------------
    def fit(self, frame, tracks):

        for t in tracks:
            if not self.is_player(t):
                continue

            color = extract_jersey_color(frame, t.bbox)
            if color is not None:
                self._color_buffer.append(color)

        if len(self._color_buffer) < self.min_samples:
            return False

        jersey_colors = np.array(self._color_buffer, dtype=np.float32)

        self.kmeans = KMeans(
            n_clusters=self.n_teams,
            n_init=20,
            random_state=42
        ).fit(jersey_colors)

        self.team_colors = self.kmeans.cluster_centers_
        return True

    # ----------------------------
    # ASSIGN — called EVERY FRAME
    # ----------------------------
    def assign(self, frame, tracks):
        if self.kmeans is None:
            return tracks

        for t in tracks:
            if not self.is_player(t):
                continue

            # 🔒 DO NOT recompute if already assigned
            if hasattr(t, "team_id") and t.team_id is not None:
                continue

            color = extract_jersey_color(frame, t.bbox)
            if color is None:
                continue

            color = np.asarray(color, dtype=np.float32).reshape(1, -1)
            label = self.kmeans.predict(color.reshape(1, -1))[0]
            t.team_id = int(label)

            """
            just for debugging team assignment:
            if not hasattr(t, "team_id"):
                print(f"Assigning team to track {t.id}")
            """

        return tracks