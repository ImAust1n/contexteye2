"""
settings.py — ContextEye Global Configuration
All tunable constants live here. No magic numbers anywhere else.
"""

# ──────────────────────────────────────────────
# Camera / Input
# ──────────────────────────────────────────────
CAMERA_URL = "http://192.168.137.201:8080/video"   # IP Webcam URL (phone server)
CAMERA_FALLBACK_INDEX = 0                          # Local webcam index if IP fails
WIFI_PORT = 8080                                   # Port IP Webcam app uses

# ──────────────────────────────────────────────
# Inference Target
# ──────────────────────────────────────────────
INFERENCE_FPS_TARGET = 10          # Target frames-per-second for the full pipeline
FRAME_SKIP_INTERVAL  = 3          # Process every Nth frame for heavy modules (depth/seg)
INPUT_RESOLUTION     = (640, 640) # Resize all frames before processing

# ──────────────────────────────────────────────
# Detection (YOLOv8s)
# ──────────────────────────────────────────────
DETECTOR_MODEL       = "yolo11s.pt"    # Model weights — auto-downloaded if absent
DETECTOR_IMGSZ       = 640            # YOLO inference resolution
DETECTOR_CONF_THRESH = 0.35           # Minimum confidence to accept a detection
DETECTOR_IOU_THRESH  = 0.45           # NMS IoU threshold
DETECTOR_DEVICE      = "auto"         # auto → uses CUDA if available, else CPU

# Classes we care about — everything else is ignored
ALLOWED_CLASSES = [
    "person", "chair", "dining table", "couch", "bed",
    "backpack", "suitcase", "bag", "bicycle", "motorbike", "car",
    "door", "stairs", "handrail", "fire hydrant", "stop sign",
    "bench", "potted plant", "tv", "laptop", "cell phone",
]

# ──────────────────────────────────────────────
# Depth Estimation (Depth Anything V2 Metric — outputs real metres)
# ──────────────────────────────────────────────
# Metric-Indoor is calibrated for 0–20m indoor/room-scale scenes.
# It outputs predicted_depth directly in metres — no conversion needed.
DEPTH_MODEL_ID      = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
DEPTH_DEVICE        = "auto"           # auto -> CUDA if available, else CPU
DEPTH_CACHE_FRAMES  = 3               # Reuse depth map for N frames before refresh
DEPTH_INFER_SIZE    = 256             # Internal inference resolution (px) -- smaller = faster

# Distance zone thresholds (in metres, based on Depth Anything V2 calibration)
DISTANCE_NEAR_M    = 1.5   # < 1.5m  → NEAR
DISTANCE_MID_M     = 4.0   # 1.5–4m  → MID
# > 4m             → FAR

WALL_AHEAD_THRESH_M = 1.2  # centre-strip avg depth below this → WALL_AHEAD

# ──────────────────────────────────────────────
# Segmentation (SegFormer-b0 ADE20K)
# ──────────────────────────────────────────────
SEGMENTOR_MODEL_ID  = "nvidia/segformer-b0-finetuned-ade-512-512"
SEGMENTOR_DEVICE    = "cpu"           # CPU to save VRAM for YOLO + Depth

# ADE20K label IDs we care about (0-indexed, see ADE20K palette)
STRUCTURAL_LABELS = {
    "wall":    0,
    "floor":   3,
    "ceiling": 5,
    "door":    14,
    "stairs":  53,
}

# ──────────────────────────────────────────────
# Spatial Reasoning
# ──────────────────────────────────────────────
POSITION_LEFT_THRESH   = 0.33   # x_center / frame_width < 0.33 → LEFT
POSITION_RIGHT_THRESH  = 0.66   # x_center / frame_width > 0.66 → RIGHT

MOTION_APPROACH_DELTA  = 0.05   # bbox area growth ratio → APPROACHING
MOTION_RECEDE_DELTA    = -0.05  # bbox area shrink ratio → RECEDING

# ──────────────────────────────────────────────
# LLM Narration
# ──────────────────────────────────────────────
LLM_MODEL_NAME  = "phi3:mini"  # Ollama model tag — must match `ollama list`
LLM_TIMEOUT_SEC = 4.0          # Max seconds to wait for LLM response
LLM_COOLDOWN_SEC = 3.0         # Min gap between LOW/MEDIUM narrations (seconds)
LLM_HOST        = "http://localhost:11434"  # Ollama local server
LLM_FAST_PATH_ENABLED = True   # If True, use rule-based for 0ms safety warnings
