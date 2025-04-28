import torch
import cv2
import numpy as np

class YOLODetector:
    def __init__(self, model_name, conf_threshold, iou_threshold, device, classes=None):
        self.device = device
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True).to(device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes

    def detect(self, frame):
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform inference
        results = self.model(rgb_frame)
        
        # Filter results based on confidence and classes
        detections = []
        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            if conf >= self.conf_threshold and (self.classes is None or int(cls) in self.classes):
                detections.append({
                    'box': box,  # [x1, y1, x2, y2]
                    'confidence': conf,
                    'class': int(cls)
                })
        return detections
