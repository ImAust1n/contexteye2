import cv2
import numpy as np
import time
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from server.detector import Detector
from server.depth import DepthEstimator
from server.segmentor import Segmentor
from server.spatial import SpatialAnalyzer
from server.hazard import classify_hazards
from server.narrator import Narrator

# ──────────────────────────────────────────────
# 1. Configuration
# ──────────────────────────────────────────────
STREAM_URL = "http://192.168.1.10:8080/video" # Updated mobile IP as needed
TARGET_FPS = 10
FRAME_SKIP = 2
PROCESS_W  = 640
PROCESS_H  = 640

# ──────────────────────────────────────────────
# 2. Main Client
# ──────────────────────────────────────────────
def main():
    print(f"📡 ContextEye IP Stream Client Starting...")
    print(f"🔗 Target: {STREAM_URL}")
    
    # 1. Initialize Pipeline
    detector  = Detector()
    depth_est = DepthEstimator()
    segmentor = Segmentor()
    analyzer  = SpatialAnalyzer(depth_estimator=depth_est)
    narrator  = Narrator()
    
    # 2. Open Stream
    cap = cv2.VideoCapture(STREAM_URL)
    
    frame_count = 0
    start_time = time.time()
    last_spoken = ""

    print("\n✅ Stream connected. Press 'ESC' to quit.")
    print("------------------------------------------")

    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("⚠️ Stream lost. Attempting to reconnect...")
                time.sleep(1)
                cap = cv2.VideoCapture(STREAM_URL)
                continue
            
            frame_count += 1
            # 3. Performance Control: Skip frames
            if frame_count % FRAME_SKIP != 0:
                continue
            
            # 4. Pipeline Execution
            proc_frame = cv2.resize(frame, (PROCESS_W, PROCESS_H))
            
            # Run Perception
            dets        = detector.detect(proc_frame)
            depth_map   = depth_est.get_depth_map(proc_frame)
            seg_results = segmentor.segment(proc_frame)
            
            spatial    = analyzer.analyze(PROCESS_W, dets, depth_map, seg_results)
            wall_data  = analyzer.detect_wall_ahead(depth_map)
            hazards    = classify_hazards(spatial, wall_data=wall_data, seg_results=seg_results)
            
            # 5. Narration (Console only for this client)
            narrator.update(hazards, wall_data=wall_data)
            current_text = narrator.last_narration
            
            if current_text != last_spoken:
                print(f"[NARRATION]: {current_text}")
                last_spoken = current_text

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
            
            if cv2.waitKey(1) & 0xFF == 27: # ESC
                break
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Done.")

if __name__ == "__main__":
    main()
