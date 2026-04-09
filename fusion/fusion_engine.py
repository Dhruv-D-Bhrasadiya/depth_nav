import numpy as np

class FusionEngine:
    def __init__(self, config):
        self.config = config['fusion']
        self.max_depth = self.config.get('max_depth', 10.0)
        self.margin = self.config.get('margin', 10)

    def process(self, frame, detections, depth_map):
        """
        Calculates median depth for each bounding box and assigns spatial position
        Returns list of objects with depth and spatial information
        """
        fused_objects = []
        h, w = depth_map.shape
        
        # Invert depth map to approximate distance
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max - depth_min > 0:
            normalized_disparity = (depth_map - depth_min) / (depth_max - depth_min)
            # Avoid division by zero
            normalized_disparity = np.clip(normalized_disparity, 0.01, 1.0)
            distance_map = 1.0 / normalized_disparity 
        else:
            distance_map = np.ones_like(depth_map) * self.max_depth
            
        distance_map = np.clip(distance_map, 0.0, self.max_depth)

        for obj in detections:
            x1, y1, x2, y2 = obj['bbox']
            
            # Apply margin to avoiding edges
            x1_m = max(0, x1 + self.margin)
            y1_m = max(0, y1 + self.margin)
            x2_m = min(w, x2 - self.margin)
            y2_m = min(h, y2 - self.margin)
            
            if x2_m <= x1_m or y2_m <= y1_m:
                median_dist = self.max_depth
            else:
                region_depths = distance_map[y1_m:y2_m, x1_m:x2_m]
                median_dist = np.median(region_depths)
                
            # Determine position (LEFT, CENTER, RIGHT)
            center_x = (x1 + x2) / 2.0
            if center_x < w * 0.33:
                pos = "LEFT"
            elif center_x > w * 0.66:
                pos = "RIGHT"
            else:
                pos = "CENTER"
                
            obj['distance'] = float(median_dist)
            obj['position'] = pos
            fused_objects.append(obj)
            
        return fused_objects, distance_map
