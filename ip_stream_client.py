import time
import sys
import os
import threading
import queue
import pyttsx3
import pythoncom

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from server.detector import Detector
from server.depth import DepthEstimator
from server.segmentor import Segmentor
from server.spatial import SpatialAnalyzer
from server.hazard import classify_hazards
from server.narrator import Narrator
from server.speaker import LocalSpeaker

from server.narration_guard import process_narration
import cv2
import numpy as np

# ──────────────────────────────────────────────
# 1. Configuration
# ──────────────────────────────────────────────
STREAM_URL = "http://192.168.137.201:8080/video" 
PROCESS_W  = 640
PROCESS_H  = 640

# ──────────────────────────────────────────────
# 2. Latency-Free Camera Thread
# ──────────────────────────────────────────────
class ThreadedCamera:
    """
    Continuously drains the OpenCV buffer in a background thread
    so the pipeline always receives the 'freshest' possible frame.
    """
    def __init__(self, url):
        self.url = url
        self.cap = cv2.VideoCapture(url)
        self.ret = False
        self.frame = None
        self.stopped = False
        
        # Start background thread
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        print(f"[Camera] Threaded reader started for {url}")

    def _update(self):
        """Background thread: Keep reading frames as fast as possible."""
        while not self.stopped:
            if not self.cap.isOpened():
                time.sleep(1)
                self.cap = cv2.VideoCapture(self.url)
                continue
                
            try:
                ret, frame = self.cap.read()
                if ret:
                    self.ret = True
                    self.frame = frame
                else:
                    self.ret = False
                    # If stream fails, wait and try to reopen
                    self.cap.release()
                    time.sleep(1)
                    self.cap = cv2.VideoCapture(self.url)
            except Exception as e:
                print(f"[Camera] Thread error: {e}")
                time.sleep(1)

    def get_frame(self):
        """Returns the absolute latest frame."""
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()
def main():
    print(f"📡 ContextEye IP Stream Client Starting...")
    print(f"🔗 Target: {STREAM_URL}")
    
    # 1. Initialize Pipeline
    detector  = Detector()
    depth_est = DepthEstimator()
    segmentor = Segmentor()
    analyzer  = SpatialAnalyzer(depth_estimator=depth_est)
    narrator  = Narrator()
    speaker   = LocalSpeaker()
    
    # 2. Initialize Camera
    camera = ThreadedCamera(STREAM_URL)
    
    frame_count = 0
    start_time  = time.time()
    last_spoken_text = ""
    last_spoken_time = 0.0

    print("\n✅ Zero-latency stream ready. Press 'ESC' to quit.")
    print("------------------------------------------")

    try:
        while True:
            loop_start = time.time()
            
            # 3. Get Freshest Frame
            ret, frame = camera.get_frame()
            
            if not ret or frame is None:
                continue
            
            frame_count += 1
            
            # 4. Pipeline Execution
            loop_timestamp = time.time() # Mark when we started processing this frame
            proc_frame = cv2.resize(frame, (640, 640))
            
            # Run Perception
            dets        = detector.detect(proc_frame)
            depth_map   = depth_est.get_depth_map(proc_frame)
            seg_results = segmentor.segment(proc_frame)
            
            spatial    = analyzer.analyze(640, dets, depth_map, seg_results)
            wall_data  = analyzer.detect_wall_ahead(depth_map)
            hazards    = classify_hazards(spatial, wall_data=wall_data, seg_results=seg_results)
            
            # 5. Intelligent Pacing: Immediate Response for Dangers
            narrator.update(hazards, wall_data=wall_data, timestamp=loop_timestamp)
            current_text = narrator.last_narration
            now = time.time()
            
            # TRIGGER LOGIC: 
            # 1. If hazard changed (Urgent or New) -> Speak Immediately
            # 2. If hazard persists (Urgent) -> Repeat every 2.5s for safety
            # 3. If info persists (Low) -> Stay silent
            is_urgent   = narrator.last_priority in ["HIGH", "MEDIUM"]
            has_changed = (current_text != last_spoken_text)
            cooldown    = 2.5 if is_urgent else 5.0
            
            if has_changed:
                # Instant trigger for urgent changes
                if is_urgent or (now - last_spoken_time) > cooldown:
                    print(f"[NARRATION]: {current_text}")
                    speaker.speak(current_text)
                    last_spoken_text = current_text
                    last_spoken_time = now
            elif is_urgent and (now - last_spoken_time) > cooldown:
                # Persistent urgent reminder
                speaker.speak(current_text)
                last_spoken_time = now

            # 6. Visualization & Bonus Overlay
            # Draw Detections
            for obj in spatial:
                box = obj["bbox"]
                label = obj["label"]
                dist = obj["distance_m"]
                zone = obj["distance_zone"]
                
                # Color based on zone
                color = (0, 0, 255) if zone == "NEAR" else (0, 255, 255) # Red for near, Yellow for mid
                
                cv2.rectangle(proc_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                cv2.putText(proc_frame, f"{label} {dist}m", (int(box[0]), int(box[1]-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw Wall/Path Info
            if wall_data.get("ahead"):
                cv2.putText(proc_frame, "WALL AHEAD", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            # FPS Counter
            current_fps = frame_count / (time.time() - start_time)
            cv2.putText(proc_frame, f"FPS: {current_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # UI Window
            cv2.imshow("ContextEye IP Stream Test", proc_frame)
            
            # 7. FPS Sync: Enforce ~3 FPS (333ms per cycle)
            loop_elapsed = time.time() - loop_start
            wait_time = max(0.01, (1.0 / 3.0) - loop_elapsed)
            time.sleep(wait_time)
            
            if cv2.waitKey(1) & 0xFF == 27: # ESC
                break
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("Done.")

if __name__ == "__main__":
    main()
