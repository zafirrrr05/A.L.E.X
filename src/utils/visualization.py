import cv2
import numpy as np

BALL_MEMORY_FRAMES = 8  # ~0.25 sec at 30fps


class BallInterpolator:
    def __init__(self, max_gap=30):
        self.max_gap = max_gap
        self.last_bbox = None
        self.last_seen = -1


    def update_from_detections(self, frame_id, detections):
        for d in detections:
            if not isinstance(d, dict):
                continue

            if d.get("class") != 0:
                continue

            bbox = d.get("bbox")
            if bbox is None:
                continue

            self.last_bbox = bbox
            self.last_seen = frame_id
            return self.last_bbox

        if self.last_bbox is not None and frame_id - self.last_seen <= self.max_gap:
            return self.last_bbox

        return None


class BallDetectionMemory:
    def __init__(self, max_gap=10):
        self.last_bbox = None
        self.last_seen = -1
        self.max_gap = max_gap

    def update(self, frame_id, detections):
        for d in detections:
            if d.get("class") == 0:
                self.last_bbox = d["bbox"]
                self.last_seen = frame_id
                return d["bbox"]

        if self.last_bbox is not None and frame_id - self.last_seen <= self.max_gap:
            return self.last_bbox

        return None


def draw_detections(frame, detections, ball_bbox=None):
    img = frame.copy()

    for d in detections:
        if isinstance(d, dict):
            bbox = d.get("bbox", None)
            cls = d.get("class", 1)
        else:
            bbox = getattr(d, "bbox", None)
            cls = getattr(d, "class", 1)

        if bbox is None or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = map(int, bbox)

        # YOLO classes:
        # 0 = ball
        # 1 = goalkeeper
        # 2 = player
        # 3 = referee
        if cls == 0:
            label = "Ball"
            color = (0, 0, 255)
        elif cls == 1:
            label = "Goalkeeper"
            color = (255, 0, 0)
        elif cls == 2:
            label = "Player"
            color = (0, 255, 0)
        else:
            label = "Referee"
            color = (255, 255, 0)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img, label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, color, 1
        )

     # 🔴 draw interpolated ball if detector missed it
    if ball_bbox is not None:
        x1, y1, x2, y2 = map(int, ball_bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            img, "Ball*",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 0), 1
        )    

    return img


def draw_tracks(frame, tracks, ball_bbox=None):

    # to find ball owner, we need to know which track (if any) is closest to the ball
    ball_owner_id = find_ball_owner(tracks, ball_bbox)

    img = frame.copy()

    # --- draw interpolated ball first ---
    if ball_bbox is not None:
        img = draw_triangle(img, ball_bbox, (0, 255, 0))

    for t in tracks:
        cls = getattr(t, "class_id", None)
        # skip ball here (already drawn)
        if cls == 0:
            continue

        bbox = getattr(t, "bbox", None)
        if bbox is None or len(bbox) != 4:
            continue

        # 1 = goalkeeper, blue ellipse
        elif cls == 1:        
            img = draw_ellipse(img, bbox, (255, 0, 0)) 
            continue

        # 2 = player, green ellipse with velocity arrow, team color if assigned
        elif cls == 2:    
            if t.team_id is not None:
                if t.team_id == 0:
                    # orange for team 0 
                    color = (0, 165, 255)
                else:
                    # light pink for team 1
                    color = (193, 182, 255)   
            else:
                # default player medium-light gray color if team unknown 
                color = (180, 180, 180)              

            img = draw_ellipse(img, bbox, color, t.id)

            # 🔴 possession arrow
            if ball_owner_id == t.id:
                img = draw_player_possession_arrow(img, bbox)
            continue

        # 3 = referee, yellow ellipse
        else:
            img = draw_ellipse(img, bbox, (0, 255, 255))   

    return img


def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_bbox_width(bbox):
    return int(bbox[2] - bbox[0])


def draw_ellipse(frame, bbox, color, track_id=None):

    y2 = int(bbox[3])
    x_center, _ = get_center_of_bbox(bbox)
    width = get_bbox_width(bbox)

    cv2.ellipse(
        frame,
        center=(x_center, y2),
        axes=(max(1, int(width)), max(1, int(0.35 * width))),
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color=tuple(int(c) for c in color),
        thickness=2,
        lineType=cv2.LINE_4
    )

    rectangle_width = 40
    rectangle_height = 20

    x1_rect = x_center - rectangle_width // 2
    x2_rect = x_center + rectangle_width // 2
    y1_rect = (y2 - rectangle_height // 2) + 15
    y2_rect = (y2 + rectangle_height // 2) + 15

    if track_id is not None:

        cv2.rectangle(
            frame,
            (int(x1_rect), int(y1_rect)),
            (int(x2_rect), int(y2_rect)),
            tuple(int(c) for c in color),
            cv2.FILLED
        )

        x1_text = x1_rect + 12
        if track_id > 99:
            x1_text -= 10

        cv2.putText(
            frame,
            f"{track_id}",
            (int(x1_text), int(y1_rect + 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

    return frame


def draw_triangle(frame, bbox, color):

    x, y = get_center_of_bbox(bbox)  # use center, not bbox[1]
    y_tip = y - 10                   # tip just above the ball
    
    triangle_points = np.array([
        [x,      y_tip],             # tip (bottom of triangle)
        [x - 10, y_tip - 20],        # top left
        [x + 10, y_tip - 20],        # top right
    ])

    cv2.drawContours(
        frame,
        [triangle_points],
        0,
        tuple(int(c) for c in color),
        cv2.FILLED
    )

    cv2.drawContours(
        frame,
        [triangle_points],
        0,
        (0, 0, 0),
        2
    )

    return frame


def draw_player_possession_arrow(frame, bbox):
    """
    Same triangle as ball, but drawn above player's head in red
    """
    x_center = int((bbox[0] + bbox[2]) / 2)
    y_top = int(bbox[1]) - 10   # above head

    triangle_points = np.array([
        [x_center,     y_top],
        [x_center - 10, y_top - 20],
        [x_center + 10, y_top - 20],
    ])

    cv2.drawContours(
        frame,
        [triangle_points],
        0,
        (0, 0, 255),   # 🔴 red
        cv2.FILLED
    )

    cv2.drawContours(
        frame,
        [triangle_points],
        0,
        (0, 0, 0),
        2
    )

    return frame


def find_ball_owner(tracks, ball_bbox, max_dist=50):
    """
    Returns track.id of player closest to the ball (or None)
    """
    if ball_bbox is None:
        return None

    bx = int((ball_bbox[0] + ball_bbox[2]) / 2)
    by = int((ball_bbox[1] + ball_bbox[3]) / 2)

    best_id = None
    best_dist = float("inf")

    for t in tracks:
        if getattr(t, "class_id", -1) != 2:  # players only
            continue

        px, py = int(t.cx), int(t.cy)
        dist = np.hypot(px - bx, py - by)

        if dist < best_dist and dist < max_dist:
            best_dist = dist
            best_id = t.id

    return best_id

