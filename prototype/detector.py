from ultralytics import YOLO

class MockDetector:
    def __init__(self):
        # We will keep the name MockDetector for the prototype structure,
        # but implement real YOLOv8 nano detection under the hood.
        # YOLOv8n automatically downloads the model weights (.pt) on the first run.
        print("Loading YOLOv8n model...")
        self.model = YOLO("yolov8n.pt")
        
        # YOLOv8 COCO class names we care about
        self.target_classes = ["person", "chair"] 

    def detect(self, frame):
        """
        Uses YOLOv8 to detect objects in the frame.
        """
        results = self.model(frame, verbose=False)
        res = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                
                # Check if it's a target class or we map others to "obstacle"
                if label not in self.target_classes:
                    if label in ["car", "motorcycle", "bus", "truck", "bicycle", "stop sign"]:
                        label = "obstacle"  # map some common objects to obstacle
                    else:
                        continue # ignore other classes to reduce noise
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w = x2 - x1
                h = y2 - y1
                
                # Ensure coordinates are within frame bounds (320x240)
                x = max(0, min(320 - w, x1))
                y = max(0, min(240 - h, y1))
                
                res.append({
                    "label": label,
                    "bbox": [int(x), int(y), int(w), int(h)]
                })
                
        return res
