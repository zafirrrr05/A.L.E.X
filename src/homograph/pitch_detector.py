import cv2
import numpy as np

class PitchDetector:
    """
    Minimal & stable pitch detector:
    Detects only:
        - center_spot
        - halfway_left
        - halfway_right
    These 3 points are reliable in broadcast footage.
    """

    def __init__(self):
        self.lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape

        lines = self.lsd.detect(gray)[0]
        if lines is None:
            return []

        lines = lines.reshape(-1, 4)

        # --- Find halfway line (strong horizontal line near middle)
        halfway = []
        for x1, y1, x2, y2 in lines:
            ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if ang < 10:    # ~horizontal
                halfway.append((x1, y1, x2, y2))

        detected = []

        if halfway:
            # center Y = median y
            ys = []
            for l in halfway:
                ys.append(l[1])
                ys.append(l[3])
            y_mid = int(np.median(ys))

            # center spot = midpoint horizontally at that y
            detected.append({"name": "center_spot", "pt": (W/2, y_mid)})
            detected.append({"name": "halfway_left", "pt": (0, y_mid)})
            detected.append({"name": "halfway_right", "pt": (W, y_mid)})

        return detected