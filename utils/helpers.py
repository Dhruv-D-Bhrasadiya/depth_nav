import time

class FPSCounter:
    def __init__(self):
        self.pTime = time.time()
        self.fps = 0

    def update(self):
        cTime = time.time()
        self.fps = 1 / (max(cTime - self.pTime, 1e-5))
        self.pTime = cTime
        return self.fps

    def draw(self, frame):
        import cv2
        cv2.putText(frame, f"FPS: {int(self.fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        return frame
