"""
spatial.py — Spatial Reasoning Layer
Takes raw outputs from Detector, Segmentor, and DepthEstimator and produces
a unified spatial description of the scene.

For each detected object it computes:
    - POSITION: LEFT | CENTER | RIGHT  (horizontal screen thirds)
    - DISTANCE_ZONE: NEAR | MID | FAR  (based on depth in metres)
    - MOTION: APPROACHING | RECEDING | STATIONARY  (bbox area delta)

Also provides a scene-level rule:
    detect_wall_ahead(depth_map) → bool

Output per object:
    {
        "label":         str,
        "position":      "LEFT" | "CENTER" | "RIGHT",
        "distance_m":    float,
        "distance_zone": "NEAR" | "MID" | "FAR",
        "motion":        "APPROACHING" | "RECEDING" | "STATIONARY",
        "bbox":          [x1, y1, x2, y2],
        "confidence":    float,
    }
"""

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    POSITION_LEFT_THRESH, POSITION_RIGHT_THRESH,
    DISTANCE_NEAR_M, DISTANCE_MID_M,
    MOTION_APPROACH_DELTA, MOTION_RECEDE_DELTA,
    WALL_AHEAD_THRESH_M,
)


class SpatialAnalyzer:
    """
    Converts raw perception outputs into human-readable spatial context.
    Maintains a history of bounding box areas to compute motion direction.
    """

    def __init__(self, depth_estimator):
        """
        Args:
            depth_estimator: instance of DepthEstimator (server/depth.py)
        """
        self.depth_estimator = depth_estimator
        # Tracking: maps object label → previous bbox area for motion detection
        self._prev_areas: dict[str, float] = {}

    # ──────────────────────────────────────────
    def analyze(
        self,
        frame_width:  int,
        detections:   list,
        depth_map:    np.ndarray,
        seg_results:  dict,
    ) -> list[dict]:
        """
        Produce spatial analysis for all detections in a single frame.

        Args:
            frame_width:  width of the processed frame in pixels.
            detections:   list of dicts from Detector.detect()
            depth_map:    float32 (H,W) array from DepthEstimator.get_depth_map()
            seg_results:  dict from Segmentor.segment()  (used for context; not
                          directly added here but callers can enrich as needed)

        Returns:
            List of spatial analysis dicts, one per detection.
        """
        spatial_objects = []

        for det in detections:
            label = det["label"]
            bbox  = det["bbox"]      # [x1, y1, x2, y2]
            conf  = det["confidence"]

            # ── Position (horizontal third) ──────────────────────────────
            x1, y1, x2, y2 = bbox
            x_center_ratio  = ((x1 + x2) / 2.0) / (frame_width + 1e-6)
            position = self._classify_position(x_center_ratio)

            # ── Distance via depth model ─────────────────────────────────
            dist_m = self.depth_estimator.get_distance(depth_map, bbox)
            zone   = self._classify_zone(dist_m)

            # ── Motion (bbox area delta from previous frame) ─────────────
            curr_area = max(1.0, (x2 - x1) * (y2 - y1))
            motion    = self._classify_motion(label, curr_area)
            # Store for next frame comparison
            self._prev_areas[label] = curr_area

            spatial_objects.append({
                "label":         label,
                "position":      position,
                "distance_m":    round(dist_m, 2),
                "distance_zone": zone,
                "motion":        motion,
                "bbox":          bbox,
                "confidence":    conf,
            })

        return spatial_objects

    # ──────────────────────────────────────────
    def detect_wall_ahead(self, depth_map: np.ndarray) -> dict:
        """
        Analyse depth in three vertical strips: LEFT, CENTER, RIGHT.
        Returns a dict indicating if an obstacle is currently blocking the path 
        ahead, and which side paths are clear.

        Args:
            depth_map: float32 (H, W) metres array.

        Returns:
            dict: {
                "ahead":       bool,  # True if CENTER is blocked
                "left_clear":  bool,  # True if LEFT is open
                "right_clear": bool,  # True if RIGHT is open
            }
        """
        H, W = depth_map.shape
        # Common vertical range: middle 60% of height (eye level to waist)
        row_start, row_end = int(H * 0.2), int(H * 0.8)
        
        # Strip boundaries (thirds)
        w3 = W // 3
        
        strips = {
            "left":   depth_map[row_start:row_end, :w3],
            "center": depth_map[row_start:row_end, w3 : 2 * w3],
            "right":  depth_map[row_start:row_end, 2 * w3 :],
        }

        # Calculate average depth for each strip
        # Note: we use np.nanmean to skip any missing depth values if present
        results = {}
        for key, strip in strips.items():
            avg = float(np.mean(strip)) if strip.size > 0 else 10.0
            # Is this strip 'clear'? (avg depth > threshold)
            results[key] = avg > WALL_AHEAD_THRESH_M

        return {
            "ahead":       not results["center"],    # 'blocked' if avg < threshold
            "left_clear":  results["left"],
            "right_clear": results["right"]
        }

    # ──────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────
    def _classify_position(self, x_ratio: float) -> str:
        """Map horizontal centre ratio → LEFT / CENTER / RIGHT."""
        if x_ratio < POSITION_LEFT_THRESH:
            return "LEFT"
        elif x_ratio > POSITION_RIGHT_THRESH:
            return "RIGHT"
        return "CENTER"

    def _classify_zone(self, dist_m: float) -> str:
        """Map metric distance → NEAR / MID / FAR."""
        if dist_m < DISTANCE_NEAR_M:
            return "NEAR"
        elif dist_m < DISTANCE_MID_M:
            return "MID"
        return "FAR"

    def _classify_motion(self, label: str, curr_area: float) -> str:
        """
        Compare current bbox area to the previous frame's area.
        Growing bbox → object approaching.  Shrinking → receding.
        """
        if label not in self._prev_areas:
            return "STATIONARY"   # first appearance — no history yet

        prev_area = self._prev_areas[label]
        delta     = (curr_area - prev_area) / (prev_area + 1e-6)

        if delta > MOTION_APPROACH_DELTA:
            return "APPROACHING"
        elif delta < MOTION_RECEDE_DELTA:
            return "RECEDING"
        return "STATIONARY"


# ──────────────────────────────────────────────────────────────────────────────
# Standalone test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, cv2
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from server.detector import Detector
    from server.depth    import DepthEstimator
    from server.camera   import ThreadedCamera
    from config.settings import CAMERA_URL

    detector  = Detector()
    depth_est = DepthEstimator()
    analyzer  = SpatialAnalyzer(depth_estimator=depth_est)

    print("[Test] Starting threaded camera...")
    cam = ThreadedCamera().start()

    print("[Test] Streaming from camera. Press 'q' to quit.")
    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            break

        dets      = detector.detect(frame)
        depth_map = depth_est.get_depth_map(frame)
        spatial   = analyzer.analyze(frame.shape[1], dets, depth_map, {})
        wall_data = analyzer.detect_wall_ahead(depth_map)

        print(f"Wall ahead: {wall_data['ahead']} (L_clear: {wall_data['left_clear']}, R_clear: {wall_data['right_clear']})")
        for obj in spatial:
            print(f"  {obj['label']} | {obj['position']} | {obj['distance_zone']} "
                  f"({obj['distance_m']:.1f}m) | {obj['motion']}")

        cv2.imshow("Spatial Analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.stop()
    cv2.destroyAllWindows()

