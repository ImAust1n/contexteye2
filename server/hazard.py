"""
hazard.py — Rule-Based Hazard Classifier
Takes the output of SpatialAnalyzer and assigns a priority level to each
object/scene condition so the narration layer knows what to speak first.

Priority levels:
    HIGH   — immediate danger, must interrupt any ongoing speech
    MEDIUM — caution warranted, speak at next opportunity
    LOW    — informational, routine heartbeat narration

Rules applied (in priority order):
    HIGH:   stairs or steps detected
    HIGH:   wall / large obstacle directly ahead (from detect_wall_ahead)
    HIGH:   any person APPROACHING in NEAR zone
    MEDIUM: door detected (navigation landmark)
    MEDIUM: chair / furniture in CENTER or NEAR zone
    MEDIUM: any object in NEAR zone (catch-all)
    LOW:    objects in MID or FAR zone
    LOW:    stationary objects anywhere
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


# Priority constants — exposed so narration layer can use them directly
HIGH   = "HIGH"
MEDIUM = "MEDIUM"
LOW    = "LOW"


def classify_hazards(
    spatial_objects: list,
    wall_data:       dict = None,
    seg_results:     dict = None,
) -> list[dict]:
    """
    Assign a hazard priority to each spatial object and add scene-level hazards.

    Args:
        spatial_objects: list of dicts from SpatialAnalyzer.analyze()
        wall_ahead:      bool from SpatialAnalyzer.detect_wall_ahead()
        seg_results:     dict from Segmentor.segment()  (checked for stairs/doors)

    Returns:
        List of hazard dicts, sorted from highest to lowest priority.
        Each dict:
            {
                "source":   "DETECTION" | "SCENE",
                "label":    str,
                "priority": HIGH | MEDIUM | LOW,
                "details":  dict   (original spatial object or scene info)
            }
    """
    hazards = []
    seg_results = seg_results or {}

    # ── Scene-level hazards (from segmentation) ──────────────────────────────

    # Stairs anywhere in the scene are HIGH priority
    if "stairs" in seg_results:
        hazards.append({
            "source":   "SCENE",
            "label":    "stairs",
            "priority": HIGH,
            "details":  {"regions": seg_results["stairs"]},
        })

    # Wall detected ahead - suggest a clear path if available
    if wall_data and wall_data.get("ahead"):
        suggestion = "Stop"
        if wall_data.get("left_clear"):
            suggestion = "Move left"
        elif wall_data.get("right_clear"):
            suggestion = "Move right"

        hazards.append({
            "source":   "SCENE",
            "label":    "wall_ahead",
            "priority": HIGH,
            "details":  {
                "message": "Large obstacle blocking forward path",
                "suggestion": suggestion
            },
        })

    # Door is a navigation landmark — always useful to mention
    if "door" in seg_results:
        hazards.append({
            "source":   "SCENE",
            "label":    "door",
            "priority": MEDIUM,
            "details":  {"regions": seg_results["door"]},
        })

    # ── Object-level hazards (from spatial analysis) ──────────────────────────

    for obj in spatial_objects:
        label  = obj["label"]
        zone   = obj["distance_zone"]         # NEAR | MID | FAR
        motion = obj["motion"]                # APPROACHING | RECEDING | STATIONARY
        pos    = obj["position"]              # LEFT | CENTER | RIGHT

        priority = _classify_object_priority(label, zone, motion, pos)

        hazards.append({
            "source":   "DETECTION",
            "label":    label,
            "priority": priority,
            "details":  obj,
        })

    # ── Sort: HIGH first, then MEDIUM, then LOW ───────────────────────────────
    _priority_rank = {HIGH: 0, MEDIUM: 1, LOW: 2}
    hazards.sort(key=lambda h: _priority_rank[h["priority"]])

    return hazards


def _classify_object_priority(
    label:  str,
    zone:   str,
    motion: str,
    pos:    str,
) -> str:
    """
    Rule engine for a single detected object.

    Args:
        label:  class name (e.g. "person", "chair")
        zone:   NEAR | MID | FAR
        motion: APPROACHING | RECEDING | STATIONARY
        pos:    LEFT | CENTER | RIGHT

    Returns:
        Priority string: HIGH | MEDIUM | LOW
    """
    # ── HIGH rules ────────────────────────────────────────────────────────────
    # Person approaching and very close
    if label == "person" and motion == "APPROACHING" and zone == "NEAR":
        return HIGH

    # Any object extremely close (catch-all safety rule)
    if zone == "NEAR" and pos == "CENTER":
        return HIGH

    # Stairs / steps (detected as YOLO label if custom-trained)
    if label in ("stairs", "step", "escalator"):
        return HIGH

    # ── MEDIUM rules ──────────────────────────────────────────────────────────
    # Door — navigation landmark
    if label == "door":
        return MEDIUM

    # Chair or furniture directly in path
    if label in ("chair", "dining table", "couch", "bench") and pos == "CENTER":
        return MEDIUM

    # Any NEAR object not already flagged HIGH
    if zone == "NEAR":
        return MEDIUM

    # Person approaching (but not yet NEAR)
    if label == "person" and motion == "APPROACHING":
        return MEDIUM

    # ── LOW rules (default) ───────────────────────────────────────────────────
    return LOW


# ──────────────────────────────────────────────────────────────────────────────
# Standalone test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulate a scene with a close approaching person and a far chair
    mock_objects = [
        {
            "label": "person", "position": "CENTER",
            "distance_m": 1.1, "distance_zone": "NEAR",
            "motion": "APPROACHING", "bbox": [100, 50, 300, 400], "confidence": 0.91
        },
        {
            "label": "chair", "position": "RIGHT",
            "distance_m": 3.5, "distance_zone": "MID",
            "motion": "STATIONARY", "bbox": [350, 200, 480, 400], "confidence": 0.74
        },
    ]
    mock_seg = {"stairs": ["CENTER"]}

    results = classify_hazards(
        spatial_objects=mock_objects,
        wall_data={"ahead": True, "left_clear": True, "right_clear": False},
        seg_results=mock_seg,
    )

    print("Hazards (sorted by priority):")
    for h in results:
        print(f"  [{h['priority']}] {h['source']} -> {h['label']}")
