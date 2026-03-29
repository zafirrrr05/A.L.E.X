import numpy as np
from collections import defaultdict


class SequenceBuilder:

    # 1 window = 50 frames, stride = 10 frames, so we get a new sequence every 10 frames with 50 frames of history each
    def __init__(self, window=50, stride=10):
        self.window = window
        self.stride = stride

    # ----------------------------
    # Track type helpers
    # ----------------------------

    # 0 = ball, 1 = goalkeeper, 2 = player, 3 = referee
    def is_ball(self, track):
        return getattr(track, "class_id", -1) == 0

    def is_referee(self, track):
        return getattr(track, "class_id", -1) == 3

    def is_player(self, track):
        return getattr(track, "class_id", -1) in (1, 2)
    

    # ----------------------------
    # Encoders
    # ----------------------------
    def encode_players(self, players):
        players = sorted(players, key=lambda x: x.id)

        feats = []
        for p in players[:11]:
            feats.append([
                p.cx, p.cy,
                p.vx, p.vy
            ])

        while len(feats) < 11:
            feats.append([0.0, 0.0, 0.0, 0.0])

        return np.asarray(feats, dtype=np.float32)

    def encode_ball(self, balls):
        if not balls:
            return np.zeros((4,), dtype=np.float32)

        b = balls[0]
        return np.array([b.cx, b.cy, b.vx, b.vy], dtype=np.float32)

    def encode_referee(self, refs):
        if not refs:
            return np.zeros((4,), dtype=np.float32)

        r = refs[0]
        return np.array([r.cx, r.cy, r.vx, r.vy], dtype=np.float32)
    

    # ----------------------------
    # Main builder
    # ----------------------------
    def build(self, track_history):

        team_tracks = {
            0: defaultdict(list),
            1: defaultdict(list)
        }
        ball_tracks = defaultdict(list)
        referee_tracks = defaultdict(list)

        # -------- collect per-frame tracks --------
        for frame_id, tracks in track_history.items():
            for t in tracks:
                if self.is_ball(t):
                    ball_tracks[frame_id].append(t)

                elif self.is_referee(t):
                    referee_tracks[frame_id].append(t)

                elif self.is_player(t) and hasattr(t, "team_id"):
                    if t.team_id in (0, 1):
                        team_tracks[t.team_id][frame_id].append(t)

        sequences = []
        frames = sorted(track_history.keys())

        # -------- sliding window --------
        for i in range(0, len(frames) - self.window, self.stride):
            window_frames = frames[i:i + self.window]

            team1_tensor = []
            team2_tensor = []
            ball_tensor = []
            referee_tensor = []

            for f in window_frames:
                team1_tensor.append(
                    self.encode_players(team_tracks[0].get(f, []))
                )
                team2_tensor.append(
                    self.encode_players(team_tracks[1].get(f, []))
                )
                ball_tensor.append(
                    self.encode_ball(ball_tracks.get(f, []))
                )
                referee_tensor.append(
                    self.encode_referee(referee_tracks.get(f, []))
                )

            sequences.append({
                "players_team1": np.stack(team1_tensor),    # (T, 11, 4)
                "players_team2": np.stack(team2_tensor),    # (T, 11, 4)
                "ball": np.stack(ball_tensor),              # (T, 4)
                "referee": np.stack(referee_tensor),        # (T, 4)
                "start_frame": window_frames[0]             # added for debugging, can be removed later, used in testing_npz
            })

        return sequences