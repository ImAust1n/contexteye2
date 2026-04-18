import cv2
from detector import MockDetector
from spatial import analyze_spatial
from motion import MotionTracker
from narrator import Narrator

def main():
    print("Initializing ContextEye Prototype...")
    
    # 1. VIDEO INPUT
    print("Welcome to ContextEye Prototype!")
    # user_input = input("Enter IP Webcam URL (e.g., http://192.168.1.100:8080/video) or press Enter to use local webcam (0): ").strip()
    
    # NOTE: The actual video stream is usually located at `/video` 
    # and dropping "https" for "http" avoids self-signed SSL handshake issues in OpenCV!
    user_input = "http://192.168.1.4:8080/video"
    
    stream_url = 0 if not user_input else user_input
    
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # Modules Initialization
    detector = MockDetector()
    tracker = MotionTracker()
    narrator = Narrator()
    
    # Resize frame target size
    WIDTH = 320
    HEIGHT = 240
    
    # --- LATENCY FIX: Threaded frame reader to always get the latest frame ---
    import threading
    import time
    
    latest_frame = None
    running = True

    def grab_frames():
        nonlocal latest_frame, running
        while running:
            ret, frame = cap.read()
            if ret:
                latest_frame = frame
            else:
                time.sleep(0.01)

    t = threading.Thread(target=grab_frames, daemon=True)
    t.start()
    
    # Wait for the first frame to arrive
    while latest_frame is None and running:
        time.sleep(0.1)

    while True:
        frame = latest_frame.copy()
        
        # Resize to 320x240 for performance
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        
        # 2. OBJECT DETECTION (BASIC)
        detections = detector.detect(frame)
        
        # 3. SPATIAL REASONING
        detections = analyze_spatial(detections, WIDTH, HEIGHT)
        
        # 4. MOTION DETECTION
        detections = tracker.track(detections)
        
        # 5, 6, 7. NARRATION AND TEXT-TO-SPEECH
        narrator.announce(detections)
        
        # 8. VISUAL DEBUG (OPTIONAL)
        for det in detections:
            x, y, w, h = det["bbox"]
            label = det["label"]
            zone = det["zone"]
            dist = det["distance"]
            motion = det.get("motion", "STATIONARY")
            
            # Highlight HIGH priority hazards in red
            is_high_hazard = (dist == "NEAR" and zone == "CENTER")
            color = (0, 0, 255) if is_high_hazard else (0, 255, 0)
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            text = f"{label} | {zone} | {dist} | {motion}"
            cv2.putText(frame, text, (x, max(10, y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv2.imshow("ContextEye", frame)
        
        # Break loop on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
