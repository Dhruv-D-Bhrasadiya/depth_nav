import cv2
import numpy as np

def draw_overlay(frame, detections, distance_map, action, reasoning):
    vis_frame = frame.copy()
    h, w = vis_frame.shape[:2]
    
    # Draw detections
    for obj in detections:
        x1, y1, x2, y2 = obj['bbox']
        cls_name = obj['class_name']
        dist = obj.get('distance', 0.0)
        risk = obj.get('risk', 0.0)
        pos = obj.get('position', 'UNKNOWN')
        
        # Color based on distance
        if dist < 1.5:
            color = (0, 0, 255) # Red
        elif dist < 3.0:
            color = (0, 165, 255) # Orange
        else:
            color = (0, 255, 0) # Green
            
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {dist:.1f}m | Risk:{risk:.1f}"
        
        # Text background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis_frame, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
        cv2.putText(vis_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    # Convert distance map to colormap for visualization
    # Invert so closer obstacles are "hotter" visually
    visual_depth = np.clip(10.0 / (distance_map + 1e-5), 0, 10.0)
    visual_depth = (visual_depth / 10.0 * 255).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(visual_depth, cv2.COLORMAP_JET)
    
    # Draw spatial regions lines
    cv2.line(vis_frame, (int(w*0.33), 0), (int(w*0.33), h), (255,255,255), 1)
    cv2.line(vis_frame, (int(w*0.66), 0), (int(w*0.66), h), (255,255,255), 1)
    
    # Action overlay panel
    panel_h = 80
    cv2.rectangle(vis_frame, (0, h - panel_h), (w, h), (0,0,0), -1)
    cv2.putText(vis_frame, f"ACT: {action}", (20, h - panel_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(vis_frame, f"RSN: {reasoning}", (20, h - panel_h + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Resize depth map and blend/put Top-Right
    depth_colormap_resized = cv2.resize(depth_colormap, (w // 3, h // 3))
    vis_frame[10:10+h//3, w - w//3 - 10 : w-10] = depth_colormap_resized
    cv2.rectangle(vis_frame, (w - w//3 - 10, 10), (w-10, 10+h//3), (255,255,255), 2)
    cv2.putText(vis_frame, "Depth Map", (w - w//3 - 5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    return vis_frame
