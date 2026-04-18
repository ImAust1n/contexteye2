import requests
import cv2
import numpy as np
import time
import threading
import queue
import sys
import os
import re

# Ensure we can import from server/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from server.detector  import Detector
from server.depth     import DepthEstimator
from server.segmentor import Segmentor
from server.spatial   import SpatialAnalyzer
from server.hazard    import classify_hazards
from server.narrator  import Narrator

# ── CONFIGURATION ──────────────────────────────
MOBILE_IP       = "192.168.137.201"
MOBILE_BASE_URL = f"http://{MOBILE_IP}:8080"
FRAME_URL       = f"{MOBILE_BASE_URL}/frame"
SPEAK_URL       = f"{MOBILE_BASE_URL}/speak"

FPS             = 3
INTERVAL        = 0.333 # Target interval between polls
MIN_SPEECH_INTERVAL = 1.5 # Fast pacing for telegraphic phrases
# ──────────────────────────────────────────────

def is_similar(text1, text2):
    """Fuzzy word-overlap check to prevent nearly identical repeats."""
    if not text1 or not text2: return False
    # Clean tags and lowercase
    t1 = set(text1.replace("[FAST]", "").lower().split())
    t2 = set(text2.replace("[FAST]", "").lower().split())
    if not t1 or not t2: return False
    
    intersection = t1.intersection(t2)
    union = t1.union(t2)
    similarity = len(intersection) / len(union)
    return similarity > 0.75

class RemoteSpeaker:
    """
    Asynchronous speaker that sends narration back to the mobile app via HTTP.
    Uses a zero-lag queue to ensure only the latest messages are transmitted.
    """
    def __init__(self, target_url):
        self.url = target_url
        self.queue = queue.Queue(maxsize=3)
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        print("[RemoteSpeaker] Remote TTS worker initialised.")

    def _worker(self):
        while True:
            item = self.queue.get()
            if item is None: break
            
            text, interrupt = item
            try:
                payload = {"text": text, "interrupt": True}
                
                # Fast timeout: drop narration rather than lagging the pipeline
                requests.post(self.url, json=payload, timeout=0.4)
                print(f"[TTS] sent: {text[:30]}...")
            except Exception:
                pass
            finally:
                self.queue.task_done()

    def speak(self, text, interrupt=False):
        """Queue a message to be sent to the phone."""
        if not text: return
        
        # Clear queue for absolute zero-lag
        while not self.queue.empty():
            try: self.queue.get_nowait()
            except queue.Empty: break
        
        try:
            self.queue.put((text, interrupt), block=False)
        except queue.Full:
            pass

def main():
    print("🎬 ContextEye Mobile Frame Client Starting...")
    print(f"📡 Target API: {MOBILE_BASE_URL}")
    
    # 1. Initialize Perception Pipeline
    detector  = Detector()
    depth_est = DepthEstimator()
    segmentor = Segmentor()
    analyzer  = SpatialAnalyzer(depth_est)
    narrator  = Narrator()
    speaker   = RemoteSpeaker(SPEAK_URL)

    last_spoken_time = 0
    last_spoken_text = ""
    speech_history   = {} # text -> timestamp
    
    print("✅ Pipeline Ready. Starting loop @ 3 FPS.")

    try:
        while True:
            loop_start = time.time()
            
            # 2. Fetch Frame (Polling)
            try:
                resp = requests.get(FRAME_URL, timeout=1.0)
                if resp.status_code != 200:
                    time.sleep(0.5)
                    continue
                
                # Decode JPEG from the response
                nparr = np.frombuffer(resp.content, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                
                print("[FRAME] received")
            except Exception as e:
                print(f"[Error] Frame request failed: {e}")
                time.sleep(0.5)
                continue

            # 3. Perception Pipeline
            loop_timestamp = time.time()
            # Resize for consistent processing speed
            proc_frame = cv2.resize(frame, (640, 640))
            
            dets        = detector.detect(proc_frame)
            depth_map   = depth_est.get_depth_map(proc_frame)
            seg_results = segmentor.segment(proc_frame)
            
            spatial    = analyzer.analyze(640, dets, depth_map, seg_results)
            wall_data  = analyzer.detect_wall_ahead(depth_map)
            hazards    = classify_hazards(spatial, wall_data=wall_data, seg_results=seg_results)
            
            print("[PIPELINE] processed")

            # 4. Narration Logic
            narrator.update(hazards, wall_data=wall_data, timestamp=loop_timestamp)
            current_text     = narrator.last_narration
            current_priority = narrator.last_priority
            
            # 5. Intelligent Trigger Flow (Anti-Spam Paced)
            now = time.time()
            elapsed_since_speech = now - last_spoken_time
            
            is_urgent   = (current_priority in ["HIGH", "MEDIUM"])
            has_changed = (current_text != last_spoken_text)
            
            # Additional Check: Is it too similar to what we just said?
            is_redundant_similar = is_similar(current_text, last_spoken_text)
            
            # Long-term History Block: Prevent saying exact same sentence for 15 seconds
            # unless it's a HIGH priority safety alert.
            is_in_recent_history = False
            last_time_seen = speech_history.get(current_text, 0)
            if (now - last_time_seen) < 15.0 and current_priority != "HIGH":
                is_in_recent_history = True
            
            # COOLDOWN logic (Consistent 2-3s Pacing)
            # HIGH: 2.5s priority warning
            # MEDIUM/LOW: 3.5s steady pulse
            if current_priority == "HIGH":
                pulse_cooldown = 2.5
            else:
                pulse_cooldown = 3.5
            
            should_speak = False
            
            if elapsed_since_speech >= MIN_SPEECH_INTERVAL:
                if has_changed and not is_redundant_similar and not is_in_recent_history:
                    # New Situation
                    should_speak = True
                elif elapsed_since_speech >= pulse_cooldown and not is_in_recent_history:
                    # Persistent reminder
                    should_speak = True
                
            if should_speak:
                # ALL-INTERRUPT: Always set interrupt=True to clear mobile app queue
                speaker.speak(current_text, interrupt=True)
                
                last_spoken_text = current_text
                last_spoken_time = now
                speech_history[current_text] = now
                
                # Cleanup old history
                if len(speech_history) > 20:
                    speech_history = {k: v for k, v in speech_history.items() if (now - v) < 20.0}
                
                print(f"[NARRATION] {current_text}")

            # 6. Show Preview (Optional Debug Window)
            cv2.imshow("ContextEye Mobile Frame Test", proc_frame)
            if cv2.waitKey(1) & 0xFF == 27: 
                break

            # 7. FPS Sync (Targeting 333ms total loop time)
            elapsed = time.time() - loop_start
            print(f"[LATENCY] {elapsed:.2f}s")
            
            wait_time = max(0.01, INTERVAL - elapsed)
            time.sleep(wait_time)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cv2.destroyAllWindows()
        print("Done.")

if __name__ == "__main__":
    main()
