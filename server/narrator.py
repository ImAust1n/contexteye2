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
    "Act as a navigation assistant for the blind. "
    "Narrate ONLY the 1-2 most critical items from the SCENE. "
    "Max 10 words. No full sentences. No chatter. Be blunt. "
    "Example: 'Person approaching center. Wall ahead.' "
    "If clear, say 'Path clear.'"
)


class Narrator:
    """
    Manages LLM narration with priority-based scheduling.
    Call update() every frame; the background thread handles Ollama inference.
    """

    def __init__(self):
        self.last_narration: str = "Initialising..."
        self.last_priority:  str = "LOW"
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
    def update(self, hazards: list, wall_data: dict = None):
        """
        Called every frame with the current hazard list.
        Decides whether to trigger a new narration based on priority and cooldown.

        Args:
            hazards:    sorted list from hazard.classify_hazards()
            wall_data:  dict from SpatialAnalyzer.detect_wall_ahead()
        """
        if not hazards and not wall_data:
            return
        
        wall_ahead = wall_data.get("ahead", False) if wall_data else False
        
        # Identify the primary concern
        primary = select_primary_hazard(hazards)
        primary_label = primary["label"] if primary else "clear"
        top_priority = primary["priority"] if primary else LOW

        # ── FAST PATH: Instant safety override ───────────────────────────────
        # If we have a HIGH hazard and Fast Path is enabled, update text immediately.
        if LLM_FAST_PATH_ENABLED and top_priority == HIGH:
            fast_text = self._rule_based_fallback(hazards, wall_data)
            with self._lock:
                # Update text but don't reset _last_spoken for the LLM context path
                self.last_narration = "[FAST] " + fast_text
                self.last_priority  = HIGH

        with self._lock:
            now = time.time()
            elapsed = now - self._last_spoken
            
            # Trigger Logic
            is_new_info = (top_priority != LOW) or (primary_label != self._last_primary_hazard)
            
            # HIGH/MEDIUM: Frequent updates + repeat if same (safety)
            if top_priority in [HIGH, MEDIUM]:
                cooldown = 2.0 if top_priority == HIGH else 3.0
                if not self._is_running and (elapsed >= cooldown or is_new_info):
                    self._trigger_narration(hazards, wall_data, top_priority)
                    self._last_primary_hazard = primary_label
                return

            # LOW: Informational; strictly deduplicate and respects cooldown
            if not self._is_running and elapsed >= LLM_COOLDOWN_SEC and is_new_info:
                self._trigger_narration(hazards, wall_data, top_priority)
                self._last_primary_hazard = primary_label

    # ─────────────────────────────────────────────────────────────────────────
    def _trigger_narration(self, hazards: list, wall_data: dict, priority: str):
        """Starts a background thread to call Ollama. Non-blocking."""
        self._is_running       = True
        self._pending_priority = priority
        thread = threading.Thread(
            target=self._run_llm,
            args=(hazards, wall_data, priority),
            daemon=True,
        )
        thread.start()

    # ─────────────────────────────────────────────────────────────────────────
    def _run_llm(self, hazards: list, wall_data: dict, priority: str):
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
        Converts the hazard list into a focused scene description for Phi-3.
        Focuses on the PRIMARY hazard to ensure conciseness.
        """
        primary = select_primary_hazard(hazards)
        if not primary and not (wall_data and wall_data.get("ahead")):
            return f"SCENE: Path clear.\n\nNarrate this for a visually impaired navigator:"

        lines = []
        if wall_data and wall_data.get("ahead"):
            suggestion = wall_data.get("suggestion", "Stop or turn.")
            lines.append(f"WALL blocking center path [HIGH]. Instruction: {suggestion}.")
        
        if primary:
            label    = primary["label"]
            priority = primary["priority"]
            det      = primary.get("details", {})
            
            if primary["source"] == "DETECTION":
                pos    = det.get("position", "")
                dist   = det.get("distance_m", "?")
                motion = det.get("motion", "")
                lines.append(f"{label.capitalize()} {dist}m {pos.lower()}, {motion.lower()} [{priority}].")
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
        if wall_data and wall_data.get("ahead"):
            suggestion = wall_data.get("suggestion", "Stop.")
            raw = f"Wall ahead. {suggestion}"
        elif not hazards:
            raw = "Path clear."
        else:
            top = hazards[0]
            label = top["label"]
            det   = top.get("details", {})
            pos   = det.get("position", "").lower()
            if top["priority"] == HIGH:
                raw = f"Warning: {label} {pos}. Stop."
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
