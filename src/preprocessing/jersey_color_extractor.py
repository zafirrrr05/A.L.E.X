import cv2
import numpy as np

def extract_jersey_color(frame, bbox):
    """
    Extract jersey color from upper torso region only
    """
    x1, y1, x2, y2 = map(int, bbox)
    h = y2 - y1

    # upper torso (avoid grass & shorts)
    y_top = y1 + int(0.15 * h)
    y_bot = y1 + int(0.45 * h)

    crop = frame[y_top:y_bot, x1:x2]

    if crop.size == 0:
        return None

    crop = cv2.resize(crop, (20, 20))
    return crop.reshape(-1, 3).mean(axis=0)