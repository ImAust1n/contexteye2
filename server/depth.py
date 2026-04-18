"""
depth.py — Depth Estimation Layer (Optimised)
Uses Depth Anything V2 Small (HuggingFace) to estimate per-pixel depth.

Performance optimisations applied:
    1. Inference runs on a small internal resolution (DEPTH_INFER_SIZE) and the
       result is upsampled back — much faster than full-res inference.
    2. Frame caching: depth map is recomputed only every DEPTH_CACHE_FRAMES
       frames; between refreshes the last map is returned instantly.
    3. float16 on GPU (half-precision) — halves memory bandwidth and compute.
    4. Batch dimension kept as (1,...) throughout to avoid squeeze/unsqueeze cost.

Output:
    get_depth_map(frame)        -> np.ndarray (H, W) float32, metres (approx)
    get_distance(depth_map, bbox) -> float  (metres, lower = closer)
"""

import cv2
import numpy as np
import torch
from PIL import Image
import sys, os, time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config.settings import DEPTH_MODEL_ID, DEPTH_DEVICE, DEPTH_CACHE_FRAMES

# ── Internal inference resolution ─────────────────────────────────────────────
# 256 px gives ~3-4× speedup vs 518 px default with minimal quality loss
# for obstacle-level navigation. Increase to 384 if accuracy matters more.
DEPTH_INFER_SIZE = 256


class DepthEstimator:
    """
    Loads Depth Anything V2 Small and exposes two public methods:
        - get_depth_map()  : full frame -> cached depth array
        - get_distance()   : bbox -> single scalar (metres)
    """

    def __init__(self):
        """Load model + processor once. Applies float16 on GPU automatically."""
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        # Resolve device
        if DEPTH_DEVICE == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(DEPTH_DEVICE)

        print(f"[Depth] Loading {DEPTH_MODEL_ID} on {self.device}...")
        self.processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
        self.model     = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID)

        # Use half-precision (float16) on GPU: halves VRAM + speeds up inference
        if self.device.type == "cuda":
            self.model = self.model.half()
        self.model.to(self.device).eval()

        # Cache state
        self._last_depth_map: np.ndarray | None = None
        self._frame_counter: int = 0
        self._last_latency_ms: float = 0.0

        print(f"[Depth] Ready on {self.device}. "
              f"Inference res: {DEPTH_INFER_SIZE}px. "
              f"Cache: every {DEPTH_CACHE_FRAMES} frames.")

    # ─────────────────────────────────────────────────────────────────────────
    def get_depth_map(self, frame: np.ndarray) -> np.ndarray:
        """
        Return a float32 (H, W) depth map in approximate metres.

        Runs inference only every DEPTH_CACHE_FRAMES calls; otherwise returns
        the cached result instantly (< 0.1 ms).

        Args:
            frame: BGR numpy array from OpenCV (any resolution).

        Returns:
            depth_map: float32 (H, W) array. Larger value = further away.
        """
        self._frame_counter += 1

        # ── Return cached map if we're within the refresh window ─────────────
        if self._last_depth_map is not None and self._frame_counter % DEPTH_CACHE_FRAMES != 1:
            return self._last_depth_map

        # ── Full inference ───────────────────────────────────────────────────
        t0 = time.perf_counter()
        orig_h, orig_w = frame.shape[:2]

        # Step 1: downscale frame to DEPTH_INFER_SIZE for fast inference
        small = cv2.resize(frame, (DEPTH_INFER_SIZE, DEPTH_INFER_SIZE),
                           interpolation=cv2.INTER_LINEAR)

        # Step 2: BGR -> RGB PIL for HuggingFace processor
        rgb    = Image.fromarray(small[..., ::-1])
        inputs = self.processor(images=rgb, return_tensors="pt")

        # Step 3: cast to float16 if on GPU
        if self.device.type == "cuda":
            inputs = {k: v.half().to(self.device) if v.dtype.is_floating_point
                      else v.to(self.device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Step 4: inference
        with torch.no_grad():
            depth_small = self.model(**inputs).predicted_depth  # (1, H', W')
            depth_small = depth_small.squeeze().float().cpu().numpy()

        # Step 5: upsample back to original frame resolution (fast — just bilinear)
        depth_full = cv2.resize(depth_small, (orig_w, orig_h),
                                interpolation=cv2.INTER_LINEAR)

        # Metric model outputs depth directly in metres — no inversion needed.
        # Just clip to a sane navigation range (0.1m minimum, 20m maximum).
        d_min, d_max = depth_full.min(), depth_full.max()
        if d_max - d_min < 1e-6:
            depth_full = np.full_like(depth_full, 5.0)  # flat/textureless scene fallback
        else:
            depth_full = np.clip(depth_full, 0.1, 20.0)

        self._last_depth_map = depth_full.astype(np.float32)
        self._last_latency_ms = (time.perf_counter() - t0) * 1000
        return self._last_depth_map

    # ─────────────────────────────────────────────────────────────────────────
    def get_distance(self, depth_map: np.ndarray, bbox: list) -> float:
        """
        Estimate distance to an object using the BOTTOM 30% of its bounding box
        and the 10th-percentile depth (ground-contact approximation).

        Args:
            depth_map: float32 (H, W) metres array from get_depth_map().
            bbox:      [x1, y1, x2, y2] pixel coords (same resolution as depth_map).

        Returns:
            float: distance in metres (lower = closer).
        """
        x1, y1, x2, y2 = map(int, bbox)
        h_box = y2 - y1

        H, W = depth_map.shape
        x1 = max(0, x1);  x2 = min(W, x2)
        y1 = max(0, y1);  y2 = min(H, y2)

        if x1 >= x2 or y1 >= y2:
            return 10.0

        # Ground-contact region: bottom 30% of the bbox height
        y_ground_start = y2 - max(1, int(h_box * 0.30))
        roi = depth_map[y_ground_start:y2, x1:x2]

        return float(np.percentile(roi, 10)) if roi.size > 0 else 10.0

    # ─────────────────────────────────────────────────────────────────────────
    @property
    def last_latency_ms(self) -> float:
        """Last measured inference latency in milliseconds (cached calls = 0)."""
        return self._last_latency_ms


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test — shows live depth map with FPS overlay
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from config.settings import CAMERA_URL
    from server.camera import ThreadedCamera

    estimator = DepthEstimator()

    print("[Test] Starting threaded camera...")
    cam = ThreadedCamera().start()

    if not cam.isOpened():
        print("[Test] No camera available -- running on blank frame.")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dm = estimator.get_depth_map(frame)
        print(f"[Test] depth shape={dm.shape}, min={dm.min():.2f}m, max={dm.max():.2f}m")
    else:
        print("[Test] Streaming from phone camera. Press 'q' to quit.")
        prev_t = time.perf_counter()
        while cam.isOpened():
            ret, frame = cam.read()
            if not ret:
                break
            depth_map = estimator.get_depth_map(frame)
            now = time.perf_counter()
            fps = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now
            vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            vis = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
            cv2.putText(vis, f"FPS: {fps:.1f}  Infer: {estimator.last_latency_ms:.0f}ms",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.imshow("Depth Map (metric metres)", vis)
            cv2.imshow("Original", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cam.stop()
        cv2.destroyAllWindows()
