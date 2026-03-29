import numpy as np

class FIFAPitchModel:
    """105 × 68 pitch geometry"""

    def __init__(self, w=105.0, h=68.0):
        self.w = w
        self.h = h
        self.cx = w / 2
        self.cy = h / 2

    def keypoints(self):
        """Return canonical labeled keypoints."""
        return {
            "center_spot": (self.cx, self.cy),
            "halfway_left": (0, self.cy),
            "halfway_right": (self.w, self.cy),

            # center circle directions
            "cc_top": (self.cx, self.cy - 9.15),
            "cc_bottom": (self.cx, self.cy + 9.15),
            "cc_left": (self.cx - 9.15, self.cy),
            "cc_right": (self.cx + 9.15, self.cy),

            # penalty spots
            "pen_L": (11, self.cy),
            "pen_R": (self.w - 11, self.cy),

            # penalty box
            "pb_L_top": (16.5, self.cy - 20.15),
            "pb_L_bottom": (16.5, self.cy + 20.15),
            "pb_R_top": (self.w - 16.5, self.cy - 20.15),
            "pb_R_bottom": (self.w - 16.5, self.cy + 20.15),
        }