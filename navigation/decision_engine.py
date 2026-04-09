import numpy as np

class DecisionEngine:
    def __init__(self, config):
        self.nav_cfg = config['navigation']
        self.left_ratio = self.nav_cfg['regions']['left_ratio']
        self.right_ratio = self.nav_cfg['regions']['right_ratio']
        self.stop_threshold = self.nav_cfg.get('stop_threshold', 3.0)
        
    def get_action(self, fused_objects, distance_map):
        h, w = distance_map.shape
        action = "MOVE_FORWARD"
        reasoning = "Path clear"
        
        # 1. Critical priority stop check based on dynamics and severely breached boundaries
        for obj in fused_objects:
            if obj['group'] == 'dynamic_obstacles' and obj['distance'] < obj['min_distance']:
                return "STOP", f"Critical: Dynamic '{obj['class_name']}' close ({obj['distance']:.2f}m)", None
            # General strict breach for large statics or unknown that are extremely close
            elif obj['distance'] < obj['min_distance'] * 0.5:
                return "STOP", f"Critical: '{obj['class_name']}' extremely close ({obj['distance']:.2f}m)", None

        # 2. Risk Aggregation per region
        region_risks = {"LEFT": 0.0, "CENTER": 0.0, "RIGHT": 0.0}
        
        # Aggregate discrete objects risks
        for obj in fused_objects:
            region_risks[obj['position']] += obj['risk']

        # Sample base environment free space risks
        h_start = int(h * 0.4)
        w_l = int(w * self.left_ratio)
        w_r = int(w * self.right_ratio)
        
        left_dist = float(np.median(distance_map[h_start:, :w_l]))
        center_dist = float(np.median(distance_map[h_start:, w_l:w_r]))
        right_dist = float(np.median(distance_map[h_start:, w_r:]))
        
        eps = 0.1
        region_risks["LEFT"] += 1.0 / (left_dist + eps)
        region_risks["CENTER"] += 1.0 / (center_dist + eps)
        region_risks["RIGHT"] += 1.0 / (right_dist + eps)
        
        safest_region = min(region_risks, key=region_risks.get)
        
        center_risk = region_risks["CENTER"]
        left_risk = region_risks["LEFT"]
        right_risk = region_risks["RIGHT"]

        # 3. Risk-Based Intelligent Logic
        if center_risk >= self.stop_threshold and left_risk >= self.stop_threshold and right_risk >= self.stop_threshold:
            action, reasoning = "STOP", "All regions are excessively high risk"
        elif center_risk > 1.0 or safest_region != "CENTER":  # 1.0 threshold for safe traveling straight
            if safest_region == "LEFT":
                action, reasoning = "TURN_LEFT", f"Center risky ({center_risk:.1f}), moving to safest: Left ({left_risk:.1f})"
            elif safest_region == "RIGHT":
                action, reasoning = "TURN_RIGHT", f"Center risky ({center_risk:.1f}), moving to safest: Right ({right_risk:.1f})"
            else:
                # Fallback, means center is safest but over threshold still
                action, reasoning = "MOVE_BACKWARD", "No safe paths forward"
        else:
            action, reasoning = "MOVE_FORWARD", f"Center is lowest risk ({center_risk:.1f})"
                
        return action, reasoning, region_risks
        
    def to_velocity_command(self, action):
        """Map logical action to ROS2-style twist command (linear_x, angular_z)"""
        velocities = {
            "MOVE_FORWARD": {"linear_x": 0.5, "angular_z": 0.0},
            "MOVE_BACKWARD": {"linear_x": -0.2, "angular_z": 0.0},
            "TURN_LEFT": {"linear_x": 0.1, "angular_z": 0.5},
            "TURN_RIGHT": {"linear_x": 0.1, "angular_z": -0.5},
            "STOP": {"linear_x": 0.0, "angular_z": 0.0}
        }
        return velocities.get(action, {"linear_x": 0.0, "angular_z": 0.0})
