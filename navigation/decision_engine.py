import numpy as np

class DecisionEngine:
    def __init__(self, config):
        self.nav_cfg = config['navigation']
        self.crit_dist = self.nav_cfg['critical_person_distance']
        self.obs_dist = self.nav_cfg['obstacle_distance']
        self.left_ratio = self.nav_cfg['regions']['left_ratio']
        self.right_ratio = self.nav_cfg['regions']['right_ratio']
        self.risk_weights = self.nav_cfg.get('risk_weights', {})
        
    def get_action(self, fused_objects, distance_map):
        h, w = distance_map.shape
        action = "MOVE_FORWARD"
        reasoning = "Path clear"
        
        # 1. Critical Stop Check
        for obj in fused_objects:
            weight = self.risk_weights.get(obj['class_id'], self.risk_weights.get('default', 1.0))
            if obj['distance'] > 0:
                obj['risk'] = (1.0 / obj['distance']) * weight
            else:
                obj['risk'] = 0.0

            if obj['class_name'] == 'person' and obj['distance'] < self.crit_dist:
                return "STOP", f"Critical: Person close ({obj['distance']:.2f}m)"

        # 2. Free Space Spatial Segmentation
        w_l = int(w * self.left_ratio)
        w_r = int(w * self.right_ratio)
        
        # Sample lower middle part of the frame for obstacle avoidance (floor level)
        h_start = int(h * 0.4)
        left_dist = np.median(distance_map[h_start:, :w_l])
        center_dist = np.median(distance_map[h_start:, w_l:w_r])
        right_dist = np.median(distance_map[h_start:, w_r:])
        
        # 3. Decision Logic based on obstacles presence in spatial regions
        is_center_blocked = center_dist < self.obs_dist
        is_left_free = left_dist > self.obs_dist
        is_right_free = right_dist > self.obs_dist
        
        # Consider specific detected objects blocking paths
        for obj in fused_objects:
            if obj['distance'] < self.obs_dist:
                if obj['position'] == "CENTER":
                    is_center_blocked = True
                elif obj['position'] == "LEFT":
                    is_left_free = False
                elif obj['position'] == "RIGHT":
                    is_right_free = False
                    
        if is_center_blocked:
            if is_left_free and is_right_free:
                # Prefer the more open path
                if left_dist > right_dist:
                    action, reasoning = "TURN_LEFT", "Center blocked, Left clearest"
                else:
                    action, reasoning = "TURN_RIGHT", "Center blocked, Right clearest"
            elif is_left_free:
                action, reasoning = "TURN_LEFT", "Center blocked, Left free"
            elif is_right_free:
                action, reasoning = "TURN_RIGHT", "Center blocked, Right free"
            else:
                action, reasoning = "MOVE_BACKWARD", "All paths blocked"
                
        return action, reasoning
        
    def to_velocity_command(self, action):
        """Map logical action to ROS2-style twist command (linear_x, angular_z)"""
        if action == "MOVE_FORWARD":
            return {"linear_x": 0.5, "angular_z": 0.0}
        elif action == "MOVE_BACKWARD":
            return {"linear_x": -0.2, "angular_z": 0.0}
        elif action == "TURN_LEFT":
            return {"linear_x": 0.1, "angular_z": 0.5}
        elif action == "TURN_RIGHT":
            return {"linear_x": 0.1, "angular_z": -0.5}
        elif action == "STOP":
            return {"linear_x": 0.0, "angular_z": 0.0}
        return {"linear_x": 0.0, "angular_z": 0.0}
