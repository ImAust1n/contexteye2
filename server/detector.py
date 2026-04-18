"""
detector.py — Object Detection Layer
Runs YOLOv8s on GPU to detect people, objects, and obstacles in each frame.

Output contract per detection:
    {
        "label":      str,           # COCO class name e.g. "person"
        "confidence": float,         # 0.0 – 1.0
        "bbox":       [x1,y1,x2,y2], # pixel coords in the INPUT frame
        "class_id":   int            # COCO class index
    }
"""

import cv2
import numpy as np
from ultralytics import YOLO
import sys
import os

# Resolve config from project root regardless of where this file is imported from
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    DETECTOR_MODEL, DETECTOR_IMGSZ, DETECTOR_CONF_THRESH,
    DETECTOR_IOU_THRESH, DETECTOR_DEVICE, ALLOWED_CLASSES,
)


class Detector:
    """
    Wraps YOLOv8s for object detection.
    Model loads once at startup and is reused across all frames.
    """

    def __init__(self):
        """Load YOLOv8s. Auto-selects GPU if available, falls back to CPU."""
        import torch
        device_cfg = DETECTOR_DEVICE
        # Resolve "auto" → "cuda" or "cpu" based on what's actually available
        if device_cfg == "auto":
            device_cfg = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[Detector] Loading {DETECTOR_MODEL} on {device_cfg}...")
        self.model  = YOLO(DETECTOR_MODEL)
        self.device = device_cfg
        dummy = np.zeros((DETECTOR_IMGSZ, DETECTOR_IMGSZ, 3), dtype=np.uint8)
        self.model(dummy, imgsz=DETECTOR_IMGSZ, device=self.device, verbose=False)
        print(f"[Detector] Ready on {self.device}. Conf threshold: {DETECTOR_CONF_THRESH}")

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run YOLOv8s inference on a single BGR frame.

        Args:
            frame: BGR numpy array (H, W, 3) from OpenCV.

        Returns:
            List of detection dicts sorted by confidence descending.
        """
        results = self.model(
            frame,
            imgsz=DETECTOR_IMGSZ,
            device=self.device,
            conf=DETECTOR_CONF_THRESH,
            iou=DETECTOR_IOU_THRESH,
            verbose=False,
        )

        detections = []
        for result in results:
            for box in result.boxes:
                label     = result.names[int(box.cls[0])]
                conf      = float(box.conf[0])
                class_id  = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Filter out classes we don't need for navigation
                if label not in ALLOWED_CLASSES:
                    continue

                detections.append({
                    "label":      label,
                    "confidence": round(conf, 3),
                    "bbox":       [x1, y1, x2, y2],
                    "class_id":   class_id,
                })

        # Highest confidence first so downstream code can prioritise easily
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections


# ──────────────────────────────────────────────────────────────────────────────
# Standalone test — run this file directly to verify detection works
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from config.settings import CAMERA_URL
    from server.camera import ThreadedCamera

    detector = Detector()

    # Try phone IP cam first, fall back to local webcam
    print("[Test] Starting threaded camera...")
    cam = ThreadedCamera().start()

    if not cam.isOpened():
        print("[Test] No camera available — running on blank frame.")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        print(f"[Test] Detections on blank frame: {detector.detect(frame)}")
    else:
        print(f"[Test] Streaming from camera. Press 'q' to quit.")
        while cam.isOpened():
            ret, frame = cam.read()
            if not ret:
                break
            dets = detector.detect(frame)
            for d in dets:
                x1, y1, x2, y2 = map(int, d["bbox"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{d['label']} {d['confidence']:.2f}",
                            (x1, max(y1 - 8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.imshow("Detector — YOLO11s", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cam.stop()
        cv2.destroyAllWindows()
