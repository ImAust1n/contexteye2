import cv2
import threading
import time
import sys
import os

# Resolve settings
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config.settings import CAMERA_URL

class ThreadedCamera:
    """
    Handles frame capture in a background thread to eliminate latency 
    caused by OpenCV's internal buffering.
    Always provides the most recent frame from the source.
    """

    def __init__(self, source=CAMERA_URL, fallback=0):
        """
        Args:
            source: URL or index for the video source.
            fallback: Backup index (usually local webcam) if source fails.
        """
        self.source = source
        self.fallback = fallback
        self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            print(f"[Camera] Could not connect to {self.source}. Falling back to index {self.fallback}...")
            self.cap = cv2.VideoCapture(self.fallback)

        if not self.cap.isOpened():
            raise RuntimeError(f"[Camera] Critical Error: Could not open any video source.")

        # Set buffer size to 1 if supported (DSHOW/MSMF, etc.)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.grabbed, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        """Start the background capture thread."""
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        """Continuously read frames from the source."""
        while True:
            if self.stopped:
                return

            grabbed, frame = self.cap.read()

            with self.lock:
                self.grabbed = grabbed
                if grabbed:
                    self.frame = frame
            
            # Tiny sleep to prevent CPU hogging if capture is somehow uncapped
            if not grabbed:
                time.sleep(0.1)

    def read(self):
        """Return the latest frame (mimics VideoCapture.read)."""
        with self.lock:
            return self.grabbed, self.frame

    def stop(self):
        """Stop the background thread and release capture."""
        self.stopped = True
        if self.cap.isOpened():
            self.cap.release()

    def isOpened(self):
        return self.cap.isOpened()

if __name__ == "__main__":
    # Test script
    print("[Camera] Starting threaded capture test...")
    cam = ThreadedCamera().start()
    
    prev_t = time.perf_counter()
    while True:
        ret, frame = cam.read()
        if not ret:
            break
            
        now = time.perf_counter()
        fps = 1.0 / (now - prev_t)
        prev_t = now
        
        cv2.putText(frame, f"Live FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Threaded Camera Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
            
    cam.stop()
    cv2.destroyAllWindows()
