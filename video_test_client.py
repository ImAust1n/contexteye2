import cv2
import numpy as np
import time
import sys
import os
import threading
import queue
import pyttsx3
import pythoncom

# ──────────────────────────────────────────────
# 1. Configuration
# ──────────────────────────────────────────────
VIDEO_PATH  = "./samples/sample3.mp4"
PROCESS_FPS = 3                # Exactly 3 frames of video time per second
DISPLAY_RES = (1280, 720)

# ──────────────────────────────────────────────
# 2. Add server folder to path so we can import modules
# ──────────────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), "server"))
from server.detector  import Detector
from server.depth     import DepthEstimator
from server.segmentor import Segmentor
from server.spatial   import SpatialAnalyzer
from server.hazard    import classify_hazards
from server.narrator  import Narrator

# ──────────────────────────────────────────────
# 3. Local TTS Engine
# ──────────────────────────────────────────────

class LocalSpeaker:
    """
    Handles local text-to-speech using pyttsx3 in a dedicated worker thread.
    This prevents the engine from hanging or crashing on Windows (SAPI5).
    """
    def __init__(self):
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        print("[Speaker] Local TTS initialised with dedicated worker.")

    def _worker(self):
        """Dedicated thread for pyttsx3 processing."""
        # MUST initialize COM for SAPI5 to work in a background thread
        pythoncom.CoInitialize()
        
        try:
            print("[Speaker] SAPI5 Worker Thread started.")
            
            while True:
                text = self.queue.get()
                if text is None: break
                
                # Clean text from technical tags
                clean_text = text.replace("[FAST]", "").strip()
                
                print(f"[Speaker] Speaking: {clean_text[:40]}...")
                try:
                    # RE-INIT engine every time to ensure fresh state on Windows
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 190) 
                    engine.setProperty('volume', 1.0)
                    
                    engine.say(clean_text)
                    engine.runAndWait()
                    
                    # Explicitly stop to clear buffers
                    engine.stop()
                    del engine  # Force cleanup
                except Exception as e:
                    print(f"[Speaker] Engine error: {e}")
                finally:
                    self.queue.task_done()
        except Exception as e:
            print(f"[Speaker] Thread critical error: {e}")
        finally:
            pythoncom.CoUninitialize()

    def speak(self, text):
        """Add text to the speech queue (Non-blocking)."""
        if not text: return
        self.queue.put(text)

# ──────────────────────────────────────────────
# 4. Visualization Helpers
# ──────────────────────────────────────────────

def draw_detections(frame, spatial_objects, wall_data):
    """Draw bounding boxes, labels and distance info."""
    for obj in spatial_objects:
        x1, y1, x2, y2 = map(int, obj["bbox"])
        label = obj["label"]
        dist  = obj["distance_m"]
        zone  = obj["distance_zone"]
        pos   = obj["position"]
        motion = obj["motion"]

        # Color based on zone
        color = (0, 0, 255) if zone == "NEAR" else (0, 255, 255) if zone == "MID" else (0, 255, 0)
        
        # Bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Label strip
        text = f"{label.upper()} | {dist:.1f}m ({zone}) | {motion}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        cv2.putText(frame, text, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Wall Ahead Indicator with suggestion
    if wall_data and wall_data.get("ahead"):
        suggestion = "Stop"
        if wall_data.get("left_clear"):
            suggestion = "Move Left"
        elif wall_data.get("right_clear"):
            suggestion = "Move Right"

        cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 255), -1)
        cv2.putText(frame, f"WALL AHEAD: {suggestion}", (120, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    # Visualise the strip boundaries
    h, w = frame.shape[:2]
    w3 = w // 3
    cv2.line(frame, (w3, 0), (w3, h), (100, 100, 100), 1)
    cv2.line(frame, (2 * w3, 0), (2 * w3, h), (100, 100, 100), 1)

# ──────────────────────────────────────────────
# 5. Main Loop
# ──────────────────────────────────────────────

def main():
    print("🎬 ContextEye Video Testing Mode Starting...")
    
    # Check if video exists
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Error: Video file not found at {VIDEO_PATH}")
        print("Please place a video file at that path or update VIDEO_PATH in the script.")
        # sys.exit(1) # Commented out so it can be verified via imports without crashing
    
    # 1. Initialize Pipeline
    detector  = Detector()
    depth_est = DepthEstimator()
    segmentor = Segmentor()
    analyzer  = SpatialAnalyzer(depth_estimator=depth_est)
    narrator  = Narrator()
    speaker   = LocalSpeaker()
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0: video_fps = 30.0  # Fallback
    
    # Process exactly 3 frames per second (Slideshow mode)
    wait_delay = 333
    frame_skip = max(1, int(video_fps / PROCESS_FPS))

    paused = False
    frame_idx = 0
    
    last_spoken_text = ""
    last_spoken_time = 0.0
    
    # Visual Persistence (Fixes flashing)
    last_spatial   = []
    last_wall_data = {"ahead": False, "left_clear": True, "right_clear": True}
    last_seg       = {}
    
    print(f"✅ Video loaded: {VIDEO_PATH}")
    print("Controls: 'p'=Pause, 'r'=Restart, 'q' or 'ESC'=Quit")

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    # Loop video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                frame_idx += 1
                
                # Resize for pipeline consistency
                proc_frame = cv2.resize(frame, (640, 640))
                
                # 2. Run Perception (on every frame we read in this mode)
                dets        = detector.detect(proc_frame)
                depth_map   = depth_est.get_depth_map(proc_frame)
                seg_results = segmentor.segment(proc_frame)
                
                spatial    = analyzer.analyze(640, dets, depth_map, seg_results)
                wall_data  = analyzer.detect_wall_ahead(depth_map)
                hazards    = classify_hazards(spatial, wall_data=wall_data, seg_results=seg_results)
                
                # Update the narrator (LLM background call)
                narrator.update(hazards, wall_data=wall_data)

                # 3. Intelligent Narration / Speaking Logic
                current_text     = narrator.last_narration
                current_priority = narrator.last_priority
                now              = time.time()
                time_since_speech = now - last_spoken_time
                
                should_speak = False
                
                # Logic: Keep speaking warnings (HIGH/MEDIUM) every 2.5s even if text is same
                if current_priority in ["HIGH", "MEDIUM"]:
                    if time_since_speech >= 2.5:
                        should_speak = True
                # Logic: Informational items (LOW) speak once and wait 4s for next different item
                else:
                    if current_text != last_spoken_text and time_since_speech >= 4.0:
                        should_speak = True

                if should_speak:
                    speaker.speak(current_text)
                    last_spoken_text = current_text
                    last_spoken_time = now
                    # Print for user to see in logs
                    print(f"[NARRATION] ({current_priority}): {current_text}")

                # 4. Draw & Display (Solid visuals)
                draw_detections(proc_frame, spatial, wall_data)
                
                # Display Subtitles
                cv2.rectangle(proc_frame, (0, 600), (640, 640), (0, 0, 0), -1)
                cv2.putText(proc_frame, narrator.last_narration, (20, 625),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                cv2.imshow("ContextEye Video Test Client", proc_frame)

                # Skip to next frame in video stream
                frame_idx += frame_skip
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            # 5. UI Controls (synced to slideshow speed)
            key = cv2.waitKey(wait_delay) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('p'):
                paused = not paused
                print("[System] Paused" if paused else "[System] Resumed")
            elif key == ord('r'):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                print("[System] Video Restarted")

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
