import torch
from ultralytics import YOLO
import numpy as np


class PlayerBallDetector:

    BALL_CLASS = 0

    def __init__(self, model_path="football_centric_trained_model/best.pt"):
        self.model = YOLO(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # to avoid issues with CUDA on some systems, we explicitly check and print the device info here
        print("[System] CUDA available:", torch.cuda.is_available())

        # force model to device once
        if self.device == "cuda":
            self.model.to("cuda")
            print(f"[Detector] Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("[Detector] Using CPU")

    def detect(self, frame):
        """
        Model classes: {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}
        Ball uses a much lower confidence threshold since it's small and fast-moving.
        """

        # --- pass for players / goalkeeper / referee (normal confidence) ---
        results_players = self.model(
            frame,
            device=self.device,
            verbose=False,
            conf=0.25,
            classes=[1, 2, 3],      # goalkeeper, player, referee
        )[0]

        # --- pass for ball (low confidence) ---
        results_ball = self.model(
            frame,
            device=self.device,
            verbose=False,
            conf=0.05,              # accept weak ball detections
            iou=0.3,                # looser NMS so blurred ball isn't suppressed
            classes=[0],            # ball only
            imgsz=1280,             # higher res — ball is tiny
        )[0]

        detections = []

        for results in (results_players, results_ball):
            if results.boxes is None:
                continue

            boxes   = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            confs   = results.boxes.conf.cpu().numpy()

            for box, cls, conf in zip(boxes, classes, confs):
                detections.append({
                    "bbox":       box.tolist(),
                    "class":      int(cls),
                    "confidence": float(conf),
                })

        return detections