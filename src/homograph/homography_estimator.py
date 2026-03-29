from operator import add

import numpy as np
import cv2


class HomographyEstimator:
    """
    Computes H mapping pixel pitch landmarks → FIFA metric pitch.
    """

    def __init__(self, pitch_w=105, pitch_h=68):
        self.pitch_w = pitch_w
        self.pitch_h = pitch_h
        self.keyframes = {}
        self.H_all = {}

    # ---------------------------------------------------------
    # INTERNAL: map detected points → canonical pitch coords
    # ---------------------------------------------------------
    def _canonical_map(self, pts_dict):
        """
        Map detected image keypoints → FIFA metric coordinates.
        FIXED to match your PitchDetector names.
        """

        out_src = []
        out_dst = []

        def add(det_name, X, Y):
            if det_name in pts_dict:
                out_src.append(pts_dict[det_name])
                out_dst.append([X, Y])

        mid_x = self.pitch_w / 2
        mid_y = self.pitch_h / 2

        # --- FIXED NAMES TO MATCH YOUR DETECTOR ---
        add("halfway_left",  0, mid_y)
        add("halfway_right", self.pitch_w, mid_y)

        add("center_spot", mid_x, mid_y)

        # center circle (real distances)
        add("cc_top",    mid_x, mid_y - 9.15)
        add("cc_bottom", mid_x, mid_y + 9.15)
        add("cc_left",   mid_x - 9.15, mid_y)
        add("cc_right",  mid_x + 9.15, mid_y)

        return np.array(out_src, np.float32), np.array(out_dst, np.float32)

    # ---------------------------------------------------------
    # 🔥 NEW: FIX FOR YOUR ERROR
    # ---------------------------------------------------------
    def get_H_from_detected(self, model, mapper, pts_list):
        """
        Compute homography directly from detected keypoints.

        model, mapper are unused (kept for compatibility with your pipeline).
        """

        if pts_list is None or len(pts_list) == 0:
            return None

        # Convert list → dict
        try:
            pts_dict = {p["name"]: p["pt"] for p in pts_list}
        except Exception:
            return None

        src, dst = self._canonical_map(pts_dict)

        if len(src) < 4:
            return None

        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

        if H is None:
            return None

        # Normalize for stability
        if H[2, 2] != 0:
            H = H / H[2, 2]

        return H

    # ---------------------------------------------------------
    # KEYFRAME STORAGE
    # ---------------------------------------------------------
    def add_keyframe(self, fid, pts_list):
        H = self.get_H_from_detected(None, None, pts_list)
        if H is not None:
            self.keyframes[fid] = H

    # ---------------------------------------------------------
    # INTERPOLATION
    # ---------------------------------------------------------
    def interpolate(self, total_frames):
        if not self.keyframes:
            raise RuntimeError("No keyframe homographies")

        ids = sorted(self.keyframes.keys())

        for i in range(len(ids) - 1):
            a, b = ids[i], ids[i + 1]
            H1, H2 = self.keyframes[a], self.keyframes[b]

            for f in range(a, b + 1):
                t = (f - a) / max(1, (b - a))

                H_interp = (1 - t) * H1 + t * H2

                # Normalize after interpolation
                if H_interp[2, 2] != 0:
                    H_interp = H_interp / H_interp[2, 2]

                self.H_all[f] = H_interp

        # Fill before first keyframe
        for f in range(0, ids[0]):
            self.H_all[f] = self.keyframes[ids[0]]

        # Fill after last keyframe
        for f in range(ids[-1], total_frames):
            self.H_all[f] = self.keyframes[ids[-1]]

        return self.H_all

    # ---------------------------------------------------------
    # SMOOTHING
    # ---------------------------------------------------------
    def smooth(self, alpha=0.10):
        keys = sorted(self.H_all.keys())

        for i in range(1, len(keys)):
            f, prev = keys[i], keys[i - 1]

            H_smooth = alpha * self.H_all[f] + (1 - alpha) * self.H_all[prev]

            # Normalize again
            if H_smooth[2, 2] != 0:
                H_smooth = H_smooth / H_smooth[2, 2]

            self.H_all[f] = H_smooth