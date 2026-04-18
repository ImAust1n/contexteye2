"""
segmentor.py — Semantic Segmentation Layer
Uses SegFormer-b0 finetuned on ADE20K to identify structural scene elements:
    wall, floor, ceiling, door, stairs

Runs on CPU intentionally to preserve VRAM for YOLOv8s + Depth Anything V2.

Output: dict mapping structural label → list of screen regions (LEFT/CENTER/RIGHT)
Example:
    {
        "wall":   ["LEFT", "CENTER"],
        "floor":  ["CENTER", "RIGHT"],
        "stairs": ["CENTER"]
    }
"""

import numpy as np
import torch
from PIL import Image
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config.settings import SEGMENTOR_MODEL_ID, SEGMENTOR_DEVICE, STRUCTURAL_LABELS


class Segmentor:
    """
    Wraps SegFormer-b0-ADE20K for structural scene understanding.
    Identifies walls, floors, stairs, doors, and ceilings.
    Each label is mapped to which horizontal screen region it occupies.
    """

    def __init__(self):
        """Load SegFormer model and processor onto CPU."""
        print(f"[Segmentor] Loading {SEGMENTOR_MODEL_ID} on {SEGMENTOR_DEVICE}...")
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

        self.processor = SegformerImageProcessor.from_pretrained(SEGMENTOR_MODEL_ID)
        self.model     = SegformerForSemanticSegmentation.from_pretrained(SEGMENTOR_MODEL_ID)
        self.model.to(SEGMENTOR_DEVICE).eval()
        print("[Segmentor] Ready.")

    # ──────────────────────────────────────────
    def segment(self, frame: np.ndarray) -> dict:
        """
        Run SegFormer on a single BGR frame and return structural elements.

        Args:
            frame: BGR numpy array (H, W, 3).

        Returns:
            dict: {label_name: [region, ...]}  where region ∈ {"LEFT","CENTER","RIGHT"}
                  Only labels from STRUCTURAL_LABELS that occupy ≥ 5% of the frame
                  are included. Empty dict if nothing found.
        """
        H, W = frame.shape[:2]

        # Convert BGR → RGB PIL for HuggingFace
        rgb   = Image.fromarray(frame[..., ::-1])
        inputs = self.processor(images=rgb, return_tensors="pt").to(SEGMENTOR_DEVICE)

        with torch.no_grad():
            logits = self.model(**inputs).logits   # shape: (1, num_classes, H/4, W/4)

        # Argmax → class label per pixel; upsample to original resolution
        seg_map = logits.argmax(dim=1).squeeze().cpu().numpy()  # (H/4, W/4)
        import cv2
        seg_map = cv2.resize(seg_map.astype(np.uint8), (W, H),
                             interpolation=cv2.INTER_NEAREST)

        # ── Map each structural label to its screen region(s) ──
        results = {}
        total_pixels = H * W

        for label_name, class_id in STRUCTURAL_LABELS.items():
            # Boolean mask for this class
            mask = (seg_map == class_id)

            # Skip if this class is nearly absent in the frame
            coverage = mask.sum() / total_pixels
            if coverage < 0.05:
                continue

            regions = self._mask_to_regions(mask, W)
            if regions:
                results[label_name] = regions

        return results

    # ──────────────────────────────────────────
    def _mask_to_regions(self, mask: np.ndarray, frame_width: int) -> list:
        """
        Given a boolean mask, determine which screen thirds it occupies.

        Args:
            mask:        boolean (H, W) array.
            frame_width: width of the original frame in pixels.

        Returns:
            Sorted list of region strings: subset of ["LEFT","CENTER","RIGHT"]
        """
        one_third = frame_width // 3

        region_map = {
            "LEFT":   mask[:, :one_third],
            "CENTER": mask[:, one_third : 2 * one_third],
            "RIGHT":  mask[:, 2 * one_third :],
        }

        # A region is considered "occupied" if ≥ 10% of its columns have mask pixels
        active = []
        for region_name, region_mask in region_map.items():
            region_pixels = region_mask.sum()
            region_total  = region_mask.size
            if region_pixels / (region_total + 1e-6) >= 0.10:
                active.append(region_name)

        return active


# ──────────────────────────────────────────────────────────────────────────────
# Standalone test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import cv2
    from config.settings import CAMERA_URL

    seg = Segmentor()

    cap = cv2.VideoCapture(CAMERA_URL)
    if not cap.isOpened():
        print(f"[Test] Could not connect to {CAMERA_URL} -- falling back to local webcam.")
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Test] No camera -- segmenting a blank grey frame.")
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        print("[Test] Result:", seg.segment(frame))
    else:
        print("[Test] Streaming from phone camera. Press 'q' to quit.")
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % 5 == 0:
                result = seg.segment(frame)
                print(f"[Seg] Frame {frame_idx}: {result}")
            frame_idx += 1
            cv2.imshow("Segmentor Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
