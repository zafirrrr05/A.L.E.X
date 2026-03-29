import cv2
import numpy as np

def overlay_space(frame, space_map):

    space = space_map
    space = (space - space.min()) / (space.max() - space.min() + 1e-6)

    heat = cv2.resize(space, (frame.shape[1], frame.shape[0]))
    heat = np.uint8(255 * heat)

    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    out = cv2.addWeighted(frame, 0.6, heat, 0.4, 0)

    return out