import numpy as np

class Projector:
    def __init__(self, H_all, pitch_w=105, pitch_h=68):
        self.H = H_all
        self.pw = pitch_w
        self.ph = pitch_h

    def _proj(self, H, x, y):
        p = np.array([x, y, 1.0], dtype=np.float32)
        o = H @ p
        if o[2] == 0:
            return None
        X = o[0] / o[2]
        Y = o[1] / o[2]

        # Important: flip vertical axis
        # because broadcast cameras invert y
        Y = self.ph - Y

        return X, Y

    def project_frame(self, fid, objects):
        # Handle both dict (per-frame H) and single matrix
        if isinstance(self.H, dict):
            H = self.H.get(fid)
        else:
            H = self.H  # single global homography

        if H is None:
            return []

        out = []
        for o in objects:
            r = self._proj(H, o["cx"], o["cy"])
            if r is None:
                continue

            X, Y = r

            if -2 <= X <= self.pw + 2 and -2 <= Y <= self.ph + 2:
                o["field_x"] = float(X)
                o["field_y"] = float(Y)
                out.append(o)

        return out