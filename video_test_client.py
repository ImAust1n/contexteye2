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
from server.speaker   import LocalSpeaker

# ──────────────────────────────────────────────
# 3. Local TTS Engine
# ──────────────────────────────────────────────

# Shared LocalSpeaker imported from server.speaker

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
                    print("🎬 Video finished.")
                    break
                
                frame_idx += 1
                
                # Resize for pipeline consistency
                proc_frame = cv2.resize(frame, (640, 640))
                
                # 2. Run Perception (on every frame we read in this mode)
                loop_timestamp = time.time()
                dets        = detector.detect(proc_frame)
                depth_map   = depth_est.get_depth_map(proc_frame)
                seg_results = segmentor.segment(proc_frame)
                
                spatial    = analyzer.analyze(640, dets, depth_map, seg_results)
                wall_data  = analyzer.detect_wall_ahead(depth_map)
                hazards    = classify_hazards(spatial, wall_data=wall_data, seg_results=seg_results)
                
                # Update the narrator (LLM background call)
                narrator.update(hazards, wall_data=wall_data, timestamp=loop_timestamp)

                # 3. Intelligent Narration / Speaking Logic (Immediate Response)
                current_text     = narrator.last_narration
                current_priority = narrator.last_priority
                now              = time.time()
                time_since_speech = now - last_spoken_time
                
                is_urgent   = current_priority in ["HIGH", "MEDIUM"]
                has_changed = (current_text != last_spoken_text)
                
                # REPETITION SHIELD (Consistent 2-3s Pacing)
                # MEDIUM/LOW: 3.0s steady pulse
                if current_priority == "HIGH":
                    cooldown = 2.0
                else:
                    cooldown = 3.0

                should_speak = False
                if has_changed:
                    # Trigger immediately for new instructions if min gap met
                    if time_since_speech >= 1.5:
                        should_speak = True
                elif time_since_speech >= cooldown:
                    # Forced periodic report (2-3s window)
                    should_speak = True

                if should_speak:
                    speaker.speak(current_text)
                    last_spoken_text = current_text
                    last_spoken_time = now
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
                
                # 5. Timing Synchronization (Prevent Drift)
                # target is 333ms per frame (3 FPS)
                loop_elapsed_ms = int((time.time() - now) * 1000)
                dynamic_delay = max(5, wait_delay - loop_elapsed_ms)
                
                # Print stats for user
                if frame_idx % 10 == 0:
                    print(f"[Sync] Latency: {loop_elapsed_ms/1000:.2f}s | FPS: {1000/max(5, loop_elapsed_ms):.1f}")
                
                key = cv2.waitKey(dynamic_delay) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('p'):
                    paused = not paused
                    print("[System] Paused" if paused else "[System] Resumed")
                elif key == ord('r'):
                    frame_idx = 0
                    print("[System] Video Restarted")
                
                # Set next position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    except KeyboardInterrupt:
        pass
    finally:
        print("🎬 Session ended.")
        speaker.speak("Navigation complete.")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
