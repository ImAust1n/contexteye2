"""
narrator.py — LLM-Powered Narration Layer (Phi-3 via Ollama)
Converts structured hazard/spatial data into natural-language audio narration.

Design principles:
    - NON-BLOCKING: LLM inference runs in a background thread.
      The main pipeline is never paused waiting for a response.
    - PRIORITY-AWARE: HIGH hazards interrupt anything in progress.
      LOW/MEDIUM narrations respect a cooldown timer.
    - CONCISE PROMPTS: We give Phi-3 only what it needs — a compact
      scene description. Responses are capped at 2 short sentences.
    - OFFLINE: All inference runs locally via Ollama HTTP API.

Usage:
    narrator = Narrator()
    narrator.update(hazards, spatial_objects, wall_ahead)
    # last_narration property carries the latest spoken text
"""

import threading
import time
import requests
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    LLM_MODEL_NAME, LLM_TIMEOUT_SEC,
    LLM_COOLDOWN_SEC, LLM_HOST,
    LLM_FAST_PATH_ENABLED,
)
from server.hazard import HIGH, MEDIUM, LOW
from server.narration_guard import (
    process_narration, select_primary_hazard, 
    clean_text, resolve_contradictions,
    enforce_structure, limit_length
)


# Simplified system prompt for faster inference
# Strictest possible instructions for brevity
_SYSTEM_PROMPT = (
    "Act as a professional high-speed navigation assistant. "
    "MANDATORY: Speak every time you detect a change or periodically (2-3s). "
    "If multiple objects are present, group them into ONE cohesive sentence. "
    "Max 10-12 words. No filler. Examples: 'Person center, chair right. Steer left.' "
    "Instruction last. Environmental check first."
)


class Narrator:
    """
    Manages LLM narration with priority-based scheduling.
    Call update() every frame; the background thread handles Ollama inference.
    """

    def __init__(self):
        self.last_narration: str = "Initialising..."
        self.last_priority:  str = "LOW"
        self._last_spoken:   float = 0.0
        self._is_running:    bool = False
        
        # State tracking for redundancy reduction
        self._last_primary_hazard: str = "none"
        self._last_primary_pos:    str = ""
        self._last_primary_dist:   float = 0.0
        self._last_fast_trigger:   float = 0.0
        
        self._lock = threading.Lock()

        # State tracking
        self._is_running   = False          # Is a LLM call in progress?
        self._last_spoken  = 0.0            # Timestamp of last completed narration
        self._pending_priority = None       # Priority of the pending narration request
        self._last_primary_hazard = None    # Label of the last primary hazard
        self._ollama_available = self._check_ollama()

        print(f"[Narrator] Using model: {LLM_MODEL_NAME} @ {LLM_HOST}")
        if not self._ollama_available:
            print("[Narrator] WARNING: Ollama not reachable. Will use rule-based fallback.")

    # ─────────────────────────────────────────────────────────────────────────
    def update(self, hazards: list, wall_data: dict = None, timestamp: float = None):
        """
        Called every frame with the current hazard list and the frame's acquisition time.
        """
        if not hazards and not wall_data:
            return
        
        now = time.time()
        frame_time = timestamp or now
        
        # If the frame is ALREADY old (pipeline lag), don't even start an LLM call
        if now - frame_time > 1.0:
            return
        
        wall_ahead = wall_data.get("ahead", False) if wall_data else False
        
        # Identify the primary concern
        primary = select_primary_hazard(hazards)
        primary_label = primary["label"] if primary else "clear"
        top_priority = primary["priority"] if primary else LOW
        # ── FAST PATH: Instant safety override ───────────────────────────────
        # If we have a HIGH hazard and Fast Path is enabled, update text immediately.
        # Enforce a 3.0s cooldown to prevent firing too frequently.
        if LLM_FAST_PATH_ENABLED and (hazards and hazards[0]["priority"] == HIGH):
            with self._lock:
                now = time.time()
                if (now - self._last_fast_trigger) > 3.0:
                    fast_text = self._rule_based_fallback(hazards, wall_data)
                    self.last_narration = "[FAST] " + fast_text
                    self.last_priority  = HIGH
                    self._last_fast_trigger = now

        # Identify the primary concern
        primary = select_primary_hazard(hazards)
        primary_label = primary["label"] if primary else "none"
        primary_pos   = primary.get("details", {}).get("position", "") if primary else ""
        primary_dist  = primary.get("details", {}).get("distance_m", 0.0) if primary else 0.0
        top_priority = hazards[0]["priority"] if hazards else LOW
        
        with self._lock:
            now = time.time()
            elapsed = now - self._last_spoken
            
            # REDUNDANCY FILTER (STRICT)
            # Trigger ONLY if:
            # 1. The hazard type changed (e.g. Wall -> Person)
            # 2. Key positional data changed (e.g. Center -> Left)
            # 3. Distance changed DRAMATICALLY (> 1.5m)
            dist_diff = abs(primary_dist - self._last_primary_dist)
            is_new_situ = (primary_label != self._last_primary_hazard) or \
                          (primary_pos != self._last_primary_pos) or \
                          (dist_diff > 1.5)
            
            # HIGH/MEDIUM: Enforce silence unless situation is truly different or time passed
            if top_priority in [HIGH, MEDIUM]:
                # HIGH: 2.5s repeated warning. MEDIUM: 5.0s pulse.
                cooldown = 2.5 if top_priority == HIGH else 5.0
                if not self._is_running and (elapsed >= cooldown or is_new_situ):
                    self._trigger_narration(hazards, wall_data, top_priority, frame_time)
                    self._last_primary_hazard = primary_label
                    self._last_primary_pos    = primary_pos
                    self._last_primary_dist   = primary_dist
                return

            # LOW: Informational; strictly deduplicate and respects cooldown
            if not self._is_running and (elapsed >= LLM_COOLDOWN_SEC and is_new_situ):
                self._trigger_narration(hazards, wall_data, top_priority, frame_time)
                self._last_primary_hazard = primary_label
                self._last_primary_pos    = primary_pos
                self._last_primary_dist   = primary_dist

    # ─────────────────────────────────────────────────────────────────────────
    def _trigger_narration(self, hazards: list, wall_data: dict, priority: str, frame_time: float):
        """Starts a background thread to call Ollama. Non-blocking."""
        self._is_running       = True
        self._pending_priority = priority
        thread = threading.Thread(
            target=self._run_llm,
            args=(hazards, wall_data, priority, frame_time),
            daemon=True,
        )
        thread.start()

    # ─────────────────────────────────────────────────────────────────────────
    def _run_llm(self, hazards: list, wall_data: dict, priority: str, frame_time: float):
        """
        Background thread: builds prompt, calls Phi-3, stores result.
        Falls back to rule-based text if Ollama is unavailable.
        """
        try:
            prompt = self._build_prompt(hazards, wall_data)

            if self._ollama_available:
                response_text = self._call_ollama(prompt)
            else:
                response_text = self._rule_based_fallback(hazards, wall_data)

            with self._lock:
                # STALE CHECK: If generation took > 1.5s, the info is outdated.
                latency = time.time() - frame_time
                if latency > 1.5:
                    print(f"[Narrator] Discarding stale {priority} narration (Late by {latency:.1f}s)")
                    self._is_running = False
                    return

                # Apply Narration Guard (Cleaning, Contradictions, Length)
                text = process_narration(response_text.strip(), hazards)
                
                if not text:
                    # If empty (broken/rejected by guard), don't update last_spoken
                    self._is_running = False
                    return

                # Prefix with urgency for HIGH priority
                if priority == HIGH:
                    if not any(kw in text.lower() for kw in ["warning", "stop", "caution"]):
                        text = "Warning. " + text

                self.last_narration = text
                self.last_priority  = priority
                self._last_spoken   = time.time()
                self._is_running    = False

            print(f"[Narrator] [{priority}] {self.last_narration}")

        except Exception as e:
            print(f"[Narrator] Error during narration: {e}")
            with self._lock:
                self._is_running = False

    # ─────────────────────────────────────────────────────────────────────────
    def _build_prompt(self, hazards: list, wall_data: dict) -> str:
        """
        Converts top 3 hazards into a dense scene description for Phi-3.
        """
        from .narration_guard import select_top_hazards
        top_hazards = select_top_hazards(hazards, limit=3)
        
        if not top_hazards and not (wall_data and wall_data.get("ahead")):
            return "SCENE: Path clear. Requesting single-sentence status update."

        lines = []
        if wall_data:
            ahead = "BLOCKED" if wall_data.get("ahead") else "CLEAR"
            left  = "CLEAR" if wall_data.get("left_clear") else "BLOCKED"
            right = "CLEAR" if wall_data.get("right_clear") else "BLOCKED"
            
            if wall_data.get("ahead"):
                lines.append(f"MAP: Forward BLOCKED. Left {left}, Right {right}. SUGGESTION: {wall_data.get('suggestion', 'Stop')}.")
            else:
                lines.append(f"MAP: Forward CLEAR. Left {left}, Right {right}.")
        
        for h in top_hazards:
            label    = h["label"]
            priority = h["priority"]
            det      = h.get("details", {})
            
            if h["source"] == "DETECTION":
                pos    = det.get("position", "")
                dist   = det.get("distance_m", "?")
                lines.append(f"{label} at {dist}m {pos.lower()} [{priority}].")
            else:
                regions = det.get("regions", [])
                region_str = ", ".join(r.lower() for r in regions) if regions else "ahead"
                lines.append(f"{label.capitalize()} detected {region_str} [{priority}].")

        scene = " ".join(lines)
        return f"SCENE: {scene}\n\nNarrate this for a visually impaired navigator:"

    # ─────────────────────────────────────────────────────────────────────────
    def _call_ollama(self, prompt: str) -> str:
        """
        Sends a prompt to Ollama's local HTTP API and returns the response text.

        Uses /api/chat endpoint with system + user messages for best quality.
        stream=False waits for the complete response before returning.
        """
        payload = {
            "model": LLM_MODEL_NAME,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            "stream": False,
            "options": {
                "temperature": 0.0,    # 0.0 for maximum consistency
                "num_predict": 25,     # Aggressively capped for short phrases
                "num_ctx": 1024,       # Smaller context for low-latency
                "num_thread": 8,
                "top_k": 20,
                "top_p": 0.9,
            },
        }
        resp = requests.post(
            f"{LLM_HOST}/api/chat",
            json=payload,
            timeout=LLM_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    # ─────────────────────────────────────────────────────────────────────────
    def _rule_based_fallback(self, hazards: list, wall_data: dict) -> str:
        """Simple rule-based narration when Ollama is unavailable."""
        raw = ""
        primary = select_primary_hazard(hazards)
        suggestion = primary.get("details", {}).get("suggestion") if primary else None
        
        if wall_data and wall_data.get("ahead"):
            wall_sugg = wall_data.get("suggestion", "Stop.")
            raw = f"Wall ahead. {wall_sugg}"
        elif not hazards:
            raw = "Path clear."
        else:
            top = hazards[0]
            label = top["label"]
            det   = top.get("details", {})
            pos   = det.get("position", "").lower()
            
            # Use the suggestion from the hazard layer if available instead of hardcoded Stop/Caution
            if suggestion:
                raw = f"{label.capitalize()} {pos}. {suggestion}."
            elif top["priority"] == HIGH:
                raw = f"Warning: {label} {pos}. Stop or turn."
            else:
                raw = f"{label.capitalize()} {pos}. Caution."
        
        return process_narration(raw, hazards)

    # ─────────────────────────────────────────────────────────────────────────
    def _check_ollama(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            resp = requests.get(f"{LLM_HOST}/api/tags", timeout=2.0)
            models = [m["name"] for m in resp.json().get("models", [])]
            available = any(LLM_MODEL_NAME in m for m in models)
            if not available:
                print(f"[Narrator] Model '{LLM_MODEL_NAME}' not found in Ollama. "
                      f"Available: {models}")
            return available
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from server.detector import Detector
    from server.depth    import DepthEstimator
    from server.spatial  import SpatialAnalyzer
    from server.hazard   import classify_hazards
    from config.settings import CAMERA_URL
    import cv2

    # Initialise all modules
    detector  = Detector()
    depth_est = DepthEstimator()
    analyzer  = SpatialAnalyzer(depth_estimator=depth_est)
    narrator  = Narrator()

    cap = cv2.VideoCapture(CAMERA_URL)
    if not cap.isOpened():
        print(f"[Test] Could not connect to {CAMERA_URL} -- falling back to local webcam.")
        cap = cv2.VideoCapture(0)

    print("[Test] Full pipeline with Phi-3 narration. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run perception pipeline
        dets      = detector.detect(frame)
        depth_map = depth_est.get_depth_map(frame)
        spatial   = analyzer.analyze(frame.shape[1], dets, depth_map, {})
        wall      = analyzer.detect_wall_ahead(depth_map)
        hazards   = classify_hazards(spatial, wall_ahead=wall)

        # Trigger narration (non-blocking)
        narrator.update(hazards, wall_data=wall_data)

        # Overlay narration text on frame
        narration_text = narrator.last_narration
        # Word-wrap for display
        words = narration_text.split()
        lines, current = [], []
        for w in words:
            current.append(w)
            if len(" ".join(current)) > 50:
                lines.append(" ".join(current[:-1]))
                current = [w]
        if current:
            lines.append(" ".join(current))

        y = 30
        for line in lines:
            cv2.putText(frame, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y += 26

        cv2.imshow("ContextEye — Phi-3 Narration", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
