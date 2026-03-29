import cv2
import numpy as np
from tqdm import tqdm

class TacticalVisualizer:
    def __init__(self, pitch_w=105, pitch_h=68, out_w=900, out_h=585):
        self.pitch_w = pitch_w
        self.pitch_h = pitch_h
        self.W = out_w
        self.H = out_h

        # scaling factor from meters → pixels
        self.sx = self.W / self.pitch_w
        self.sy = self.H / self.pitch_h

        # color scheme (BGR)
        self.col_grass_1 = (25, 140, 25)  # Darker green stripe
        self.col_grass_2 = (45, 175, 45)  # Lighter green stripe
        self.white = (255, 255, 255)
        self.thick = 2
        
        # Object colors
        self.col_A = (0, 0, 255)      # Red
        self.col_B = (255, 0, 0)      # Blue
        self.col_ball = (255, 255, 255)
        self.col_ref = (0, 255, 255)

    def _to_px(self, x, y):
        px = int(x * self.sx)
        py = int((self.pitch_h - y) * self.sy)
        return px, py

    def _draw_pitch(self):
        # 1. Create Background with Vertical Stripes
        img = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        
        num_stripes = 15 
        stripe_w_px = self.W / num_stripes
        
        for i in range(num_stripes):
            color = self.col_grass_1 if i % 2 == 0 else self.col_grass_2
            x1 = int(i * stripe_w_px)
            x2 = int((i + 1) * stripe_w_px)
            cv2.rectangle(img, (x1, 0), (x2, self.H), color, -1)

        # 2. Outer Boundary
        cv2.rectangle(img, (0, 0), (self.W - 1, self.H - 1), self.white, self.thick)

        # 3. Halfway Line & Center
        mid_x, mid_y = self.pitch_w / 2, self.pitch_h / 2
        cv2.line(img, self._to_px(mid_x, 0), self._to_px(mid_x, self.pitch_h), self.white, self.thick)
        cv2.circle(img, self._to_px(mid_x, mid_y), int(9.15 * self.sx), self.white, self.thick)
        cv2.circle(img, self._to_px(mid_x, mid_y), 2, self.white, -1) # Center Spot

        # 4. Penalty Areas, Goal Areas, and Goal Nets
        box_h, goal_h = 40.3, 18.32
        net_w, net_h = 2.0, 7.32 # Dimensions for the goal net structure outside pitch
        
        for side in [0, 1]:
            is_right = (side == 1)
            x_edge = self.pitch_w if is_right else 0
            dir = -1 if is_right else 1
            
            # Penalty Box
            cv2.rectangle(img, self._to_px(x_edge, mid_y-box_h/2), 
                          self._to_px(x_edge+(dir*16.5), mid_y+box_h/2), self.white, self.thick)
            # Goal Area
            cv2.rectangle(img, self._to_px(x_edge, mid_y-goal_h/2), 
                          self._to_px(x_edge+(dir*5.5), mid_y+goal_h/2), self.white, self.thick)
            
            # Goal Net (The white boxes extending outside the pitch in your image)
            # Note: Since the image is 900x585, we draw these slightly inside or 
            # you may need to add padding to self.W if you want them truly "outside".
            # For now, we draw them right on the edge.
            cv2.rectangle(img, self._to_px(x_edge, mid_y-net_h/2), 
                          self._to_px(x_edge+(dir*2.5), mid_y+net_h/2), self.white, self.thick)

            # Penalty Spot & Arc
            spot_x = self.pitch_w - 11 if is_right else 11
            cv2.circle(img, self._to_px(spot_x, mid_y), 2, self.white, -1)
            start, end = (127, 233) if is_right else (-53, 53)
            cv2.ellipse(img, self._to_px(spot_x, mid_y), (int(9.15*self.sx), int(9.15*self.sy)), 
                        0, start, end, self.white, self.thick)

        # 5. Corner Arcs
        cr = int(1 * self.sx)
        cv2.ellipse(img, (0, 0), (cr, cr), 0, 0, 90, self.white, self.thick)
        cv2.ellipse(img, (0, self.H), (cr, cr), 0, 270, 360, self.white, self.thick)
        cv2.ellipse(img, (self.W, 0), (cr, cr), 0, 90, 180, self.white, self.thick)
        cv2.ellipse(img, (self.W, self.H), (cr, cr), 0, 180, 270, self.white, self.thick)

        return img

    def draw_frame(self, objs):
        frame = self._draw_pitch()
        for o in objs:
            if "field_x" not in o: 
                continue

            # NEW SAFETY CHECK
            if not (0 <= o["field_x"] <= self.pitch_w and 0 <= o["field_y"] <= self.pitch_h):
                continue
            
            px, py = self._to_px(o["field_x"], o["field_y"])
            cls, team = o.get("class", None), o.get("team", "unknown")

            if cls == 2: # Players
                color = self.col_A if team == "A" else self.col_B
                cv2.circle(frame, (px, py), 8, color, -1)
                cv2.circle(frame, (px, py), 9, (255, 255, 255), 1) # Small outline
            elif cls == 0: # Ball
                cv2.circle(frame, (px, py), 5, self.col_ball, -1)
        return frame

    def render_video(self, projected, output_path, fps=25):
        print(f"[Visualizer] Rendering to: {output_path}")
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (self.W, self.H))
        for f in tqdm(range(len(projected)), desc="Rendering"):
            frame = self.draw_frame(projected.get(f, []))
            writer.write(frame)
        writer.release()