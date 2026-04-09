import cv2
from ultralytics import YOLO
import torch

class ObjectDetector:
    def __init__(self, config):
        self.config = config['models']['yolo']
        self.device = "cuda" if torch.cuda.is_available() and config['system']['use_gpu'] else "cpu"
        self.model = YOLO(self.config['weights'])
        self.model.to(self.device)
        self.conf_thresh = self.config['confidence_threshold']
        self.classes = self.config['classes']

    def detect(self, frame):
        # Run inference
        results = self.model(frame, conf=self.conf_thresh, classes=self.classes, verbose=False)
        boxes_list = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                boxes_list.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "class_id": cls,
                    "class_name": self.model.names[cls],
                    "confidence": conf
                })
        return boxes_list
