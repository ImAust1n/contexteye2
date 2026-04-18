class MotionTracker:
    def __init__(self):
        # Dictionary to track last area of objects (using label+zone as mock identifier ID)
        self.history = {}

    def track(self, detections):
        """
        Track objects across frames using bounding box size.
        If area increases -> APPROACHING
        If area decreases -> RECEDING
        Else -> STATIONARY
        """
        for det in detections:
            label = det["label"]
            x, y, w, h = det["bbox"]
            area = w * h
            
            # Create a mock ID. In reality, we'd use a real tracker like SORT or DeepSORT.
            obj_id = f"{label}_{det.get('zone', 'unknown')}"
            
            if obj_id in self.history:
                last_area = self.history[obj_id]
                area_diff = area - last_area
                
                # Check for significant difference (e.g. 5%)
                if area_diff > 0.05 * last_area:
                    det["motion"] = "APPROACHING"
                elif area_diff < -0.05 * last_area:
                    det["motion"] = "RECEDING"
                else:
                    det["motion"] = "STATIONARY"
            else:
                det["motion"] = "STATIONARY"
                
            self.history[obj_id] = area
            
        return detections
