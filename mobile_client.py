import cv2
import numpy as np
import requests
import threading
import time
import sys
import os

# ──────────────────────────────────────────────
# 1. Configuration 
# ──────────────────────────────────────────────
MOBILE_IP = "192.168.1.100"  # <-- UPDATE THIS TO YOUR MOBILE IP
PORT = 8080

FRAME_URL = f"http://{MOBILE_IP}:{PORT}/frame"
SPEAK_URL = f"http://{MOBILE_IP}:{PORT}/speak"

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
# 3. Mobile Adapters
# ──────────────────────────────────────────────

class MobileCamera:
    """
    Fetches frames from mobile app via HTTP and makes them available 
    as OpenCV matrices in a thread-safe way.
    """
    def __init__(self, url):
        self.url = url
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        
        # Initial frame to avoid None errors
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Start capture thread
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        print(f"[Mobile] Camera initialised. Fetching from {url}...")

    def _update(self):
        while self.running:
            try:
                # HTTP GET for the latest JPEG frame
                response = requests.get(self.url, timeout=2.0)
                if response.status_code == 200:
                    # Convert JPEG bytes to numpy array
                    array = np.frombuffer(response.content, dtype=np.uint8)
                    decoded = cv2.imdecode(array, cv2.IMREAD_COLOR)
                    
                    if decoded is not None:
                        with self.lock:
                            self.frame = decoded
                else:
                    print(f"[Mobile] Failed to fetch frame (HTTP {response.status_code})")
            except Exception as e:
                print(f"[Mobile] Camera fetch error: {e}")
                time.sleep(1.0)  # Wait before retry
            
            # Target ~10-15 FPS to avoid saturating mobile bandwidth
            time.sleep(0.05)

    def get_frame(self):
        with self.lock:
            return self.frame.copy()

    def stop(self):
        self.running = False


def send_tts(text):
    """
    Non-blocking HTTP POST to mobile TTS endpoint.
    Spawns a one-off thread for the request.
    """
    if not text or text.startswith("[FAST]"):
        # We handle fast path differently or skip prefix
        text = text.replace("[FAST]", "").strip()

    def _post():
        try:
            payload = {"text": text}
            headers = {"Content-Type": "application/json"}
            requests.post(SPEAK_URL, json=payload, headers=headers, timeout=2.0)
            print(f"[Mobile] Speech sent: {text}")
        except Exception as e:
            # Silent fail for network issues per constraints
            pass

    threading.Thread(target=_post, daemon=True).start()


# ──────────────────────────────────────────────
# 4. Main Loop
# ──────────────────────────────────────────────

def main():
    print("🚀 ContextEye Mobile Client Starting...")
    
    # 1. Initialize Pipeline Modules
    detector  = Detector()
    depth_est = DepthEstimator()
    segmentor = Segmentor()
    analyzer  = SpatialAnalyzer(depth_estimator=depth_est)
    narrator  = Narrator()
    
    # 2. Start Mobile Camera
    camera = MobileCamera(FRAME_URL)
    
    last_processed_narration = ""
    print("✅ Connected to mobile. Processing frames...")

    try:
        while True:
            t0 = time.perf_counter()
            
            # 1. Get latest frame
            frame = camera.get_frame()
            if frame is None:
                continue

            # 2. Run Perception
            # YOLO detection
            dets = detector.detect(frame)
            
            # Metric Depth Estimation
            depth_map = depth_est.get_depth_map(frame)
            
            # Semantic Segmentation (structural)
            seg_results = segmentor.segment(frame)
            
            # Spatial Analysis (aggregator)
            spatial = analyzer.analyze(frame.shape[1], dets, depth_map, seg_results)
            
            # Scene-level checks
            wall_data = analyzer.detect_wall_ahead(depth_map)
            
            # 3. Hazard Priority Engine
            hazards = classify_hazards(spatial, wall_data=wall_data, seg_results=seg_results)

            # 4. Narration (updates narrator.last_narration in background)
            narrator.update(hazards, wall_data=wall_data)

            # 5. Check for NEW narration text to send to mobile
            current_narration = narrator.last_narration
            if current_narration != last_processed_narration:
                send_tts(current_narration)
                last_processed_narration = current_narration

            # Benchmarking / Optional logging
            dt = (time.perf_counter() - t0) * 1000
            # print(f"[Pipeline] Step: {dt:.0f}ms")

            # Optional: Show locally for debugging if screen available
            cv2.imshow("Mobile Client Debug", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[System] Shutting down...")
    finally:
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
