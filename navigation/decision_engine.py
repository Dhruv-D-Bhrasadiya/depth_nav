import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class DecisionEngine:
    def __init__(self, config):
        self.nav_cfg = config['navigation']
        self.left_ratio = self.nav_cfg['regions']['left_ratio']
        self.right_ratio = self.nav_cfg['regions']['right_ratio']
        self.stop_threshold = self.nav_cfg.get('stop_threshold', 3.0)
        
        # Fuzzy Logic Setup
        self.setup_fuzzy_system()
        
    def setup_fuzzy_system(self):
        # Antecedents (inputs)
        self.center_risk = ctrl.Antecedent(np.arange(0, 5, 0.1), 'center_risk')
        self.left_risk = ctrl.Antecedent(np.arange(0, 5, 0.1), 'left_risk')
        self.right_risk = ctrl.Antecedent(np.arange(0, 5, 0.1), 'right_risk')
        
        # Consequent (output)
        self.action = ctrl.Consequent(np.arange(0, 4, 1), 'action')
        
        # Membership functions
        self.center_risk['low'] = fuzz.trimf(self.center_risk.universe, [0, 0, 1])
        self.center_risk['medium'] = fuzz.trimf(self.center_risk.universe, [0.5, 1.5, 2.5])
        self.center_risk['high'] = fuzz.trimf(self.center_risk.universe, [2, 3, 5])
        
        self.left_risk['low'] = fuzz.trimf(self.left_risk.universe, [0, 0, 1])
        self.left_risk['medium'] = fuzz.trimf(self.left_risk.universe, [0.5, 1.5, 2.5])
        self.left_risk['high'] = fuzz.trimf(self.left_risk.universe, [2, 3, 5])
        
        self.right_risk['low'] = fuzz.trimf(self.right_risk.universe, [0, 0, 1])
        self.right_risk['medium'] = fuzz.trimf(self.right_risk.universe, [0.5, 1.5, 2.5])
        self.right_risk['high'] = fuzz.trimf(self.right_risk.universe, [2, 3, 5])
        
        self.action['stop'] = fuzz.trimf(self.action.universe, [0, 0, 0])
        self.action['turn_left'] = fuzz.trimf(self.action.universe, [1, 1, 1])
        self.action['turn_right'] = fuzz.trimf(self.action.universe, [2, 2, 2])
        self.action['move_forward'] = fuzz.trimf(self.action.universe, [3, 3, 3])
        
        # Rules
        rule1 = ctrl.Rule(self.center_risk['high'], self.action['stop'])
        rule2 = ctrl.Rule(self.center_risk['medium'] & self.left_risk['low'], self.action['turn_left'])
        rule3 = ctrl.Rule(self.center_risk['medium'] & self.right_risk['low'], self.action['turn_right'])
        rule4 = ctrl.Rule(self.center_risk['low'] & self.left_risk['low'] & self.right_risk['low'], self.action['move_forward'])
        rule5 = ctrl.Rule(self.center_risk['medium'] & self.left_risk['high'] & self.right_risk['high'], self.action['stop'])
        
        # Control system
        self.action_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
        self.action_sim = ctrl.ControlSystemSimulation(self.action_ctrl)
        
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
        
        safest_region = min(region_risks, key=region_risks.get)
        
        center_risk = region_risks["CENTER"]
        left_risk = region_risks["LEFT"]
        right_risk = region_risks["RIGHT"]

        # Fuzzy Logic Decision Making
        self.action_sim.input['center_risk'] = center_risk
        self.action_sim.input['left_risk'] = left_risk
        self.action_sim.input['right_risk'] = right_risk
        
        try:
            self.action_sim.compute()
            action_value = self.action_sim.output['action']
            
            # Defuzzify to get action
            if action_value < 0.5:
                action = "STOP"
                reasoning = "Fuzzy: High risk detected"
            elif action_value < 1.5:
                action = "TURN_LEFT"
                reasoning = "Fuzzy: Turning left for safety"
            elif action_value < 2.5:
                action = "TURN_RIGHT"
                reasoning = "Fuzzy: Turning right for safety"
            else:
                action = "MOVE_FORWARD"
                reasoning = "Fuzzy: Path clear to move forward"
        except:
            # Fallback to original logic if fuzzy fails
            if center_risk >= self.stop_threshold and left_risk >= self.stop_threshold and right_risk >= self.stop_threshold:
                action, reasoning = "STOP", "All regions are excessively high risk"
            elif center_risk > 1.0 or safest_region != "CENTER":
                if safest_region == "LEFT":
                    action, reasoning = "TURN_LEFT", f"Center risky ({center_risk:.1f}), moving to safest: Left ({left_risk:.1f})"
                elif safest_region == "RIGHT":
                    action, reasoning = "TURN_RIGHT", f"Center risky ({center_risk:.1f}), moving to safest: Right ({right_risk:.1f})"
                else:
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
