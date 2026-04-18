def analyze_spatial(detections, frame_width, frame_height):
    """
    Estimates the zone and distance of objects based on bounding boxes.
    """
    for det in detections:
        x, y, w, h = det["bbox"]
        center_x = x + w / 2
        bottom_y = y + h

        # Determine Zone (LEFT, CENTER, RIGHT)
        third = frame_width / 3
        if center_x < third:
            det["zone"] = "LEFT"
        elif center_x < 2 * third:
            det["zone"] = "CENTER"
        else:
            det["zone"] = "RIGHT"
        
        # Estimate Distance (NEAR, MID, FAR)
        # Using bottom_y or area. Bottom Y is effective: the lower the object's bottom edge, the closer it probably is.
        if bottom_y > 0.8 * frame_height:
            det["distance"] = "NEAR"
        elif bottom_y > 0.4 * frame_height:
            det["distance"] = "MID"
        else:
            det["distance"] = "FAR"
            
    return detections
