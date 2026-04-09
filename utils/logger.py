import csv
import os
import time

class SimulationLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Initialize header if it does not exist
        write_header = not os.path.exists(filepath)
        self.file = open(filepath, 'a', newline='')
        self.writer = csv.writer(self.file)
        
        if write_header:
            self.writer.writerow([
                "timestamp", 
                "frame_num", 
                "action", 
                "reasoning", 
                "detected_objects"
            ])
            
    def log(self, frame_idx, action, reasoning, fused_objects):
        # Format object lists nicely into strings
        obj_strings = [
            f"{o['class_name']}({o['group']} | d:{o['distance']:.2f}m | r:{o['risk']:.2f})" 
            for o in fused_objects
        ]
        
        objs_str = "; ".join(obj_strings) if obj_strings else "None"
        
        self.writer.writerow([
            time.time(), 
            frame_idx, 
            action, 
            reasoning, 
            objs_str
        ])
        
    def close(self):
        if self.file and not self.file.closed:
            self.file.close()
