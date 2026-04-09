import os
import sys
import yaml
import cv2
import argparse
import time

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.detection import ObjectDetector
from models.depth import DepthEstimator
from fusion.fusion_engine import FusionEngine
from navigation.decision_engine import DecisionEngine
from utils.visualization import draw_overlay
from utils.helpers import FPSCounter

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Depth-Aware Robot Navigation")
    parser.add_argument("--source", type=str, default="0", help="Video source: camera index (0) or path to video file")
    parser.add_argument("--config", type=str, default="../configs/config.yaml", help="Path to config file")
    return parser.parse_args()

def main():
    args = parse_args()
    
    config_path = os.path.join(os.path.dirname(__file__), args.config)
    config = load_config(config_path)

    print("[INFO] Initializing Depth Estimator (MiDaS)...")
    depth_estimator = DepthEstimator(config)
    
    print("[INFO] Initializing Object Detector (YOLOv8)...")
    detector = ObjectDetector(config)
    
    print("[INFO] Initializing Fusion Engine...")
    fusion = FusionEngine(config)
    
    print("[INFO] Initializing Decision Engine...")
    decision = DecisionEngine(config)
    
    fps_counter = FPSCounter()
    
    # Source Setup
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source {source}")
        return

    print("[INFO] Starting pipeline... Press 'q' to quit.")
    
    # For evaluation metrics
    frames_processed = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video stream.")
            break
            
        frame = cv2.resize(frame, (640, 480))
        
        # 1. Inference: Detection & Depth
        detections = detector.detect(frame)
        depth_map = depth_estimator.estimate_depth(frame)
        
        # 2. Sensor Fusion
        fused_objects, distance_map = fusion.process(frame, detections, depth_map)
        
        # 3. Decision Making
        action, reasoning = decision.get_action(fused_objects, distance_map)
        cmd_vel = decision.to_velocity_command(action)
        
        # 4. Visualization
        vis_frame = draw_overlay(frame, fused_objects, distance_map, action, reasoning)
        fps_counter.update()
        vis_frame = fps_counter.draw(vis_frame)
        
        # Overlay cmd_vel
        cv2.putText(vis_frame, f"Vel: v={cmd_vel['linear_x']:.2f}, w={cmd_vel['angular_z']:.2f}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow("Depth-Aware Robot Navigation", vis_frame)
        
        frames_processed += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Print basic evaluation
    total_time = time.time() - start_time
    if total_time > 0:
        print("\n--- Pipeline Evaluation ---")
        print(f"Total frames processed: {frames_processed}")
        print(f"Average FPS: {frames_processed / total_time:.2f}")
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
