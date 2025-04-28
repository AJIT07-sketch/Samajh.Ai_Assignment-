import unittest
import cv2
import torch
import numpy as np
import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.detector.detection_model import YOLODetector, Detection


class TestDetection(unittest.TestCase):
    """Test cases for Detection class."""
    
    def setUp(self):
        """Set up test case."""
        self.bbox = [10, 20, 30, 40]  # [x1, y1, x2, y2]
        self.confidence = 0.85
        self.class_id = 0
        self.class_name = "person"
        self.detection = Detection(self.bbox, self.confidence, self.class_id, self.class_name)
        
    def test_detection_initialization(self):
        """Test Detection initialization."""
        self.assertEqual(self.detection.bbox, self.bbox)
        self.assertEqual(self.detection.confidence, self.confidence)
        self.assertEqual(self.detection.class_id, self.class_id)
        self.assertEqual(self.detection.class_name, self.class_name)
        self.assertIsNone(self.detection.track_id)
        
    def test_detection_centroid(self):
        """Test Detection centroid calculation."""
        expected_centroid = ((self.bbox[0] + self.bbox[2]) / 2, 
                            (self.bbox[1] + self.bbox[3]) / 2)
        self.assertEqual(self.detection.centroid, expected_centroid)


class TestYOLODetector(unittest.TestCase):
    """Test cases for YOLODetector class."""
    
    def setUp(self):
        """Set up test case."""
        # Skip tests if no GPU available and running on CI
        if not torch.cuda.is_available() and os.environ.get('CI') == 'true':
            self.skipTest("Skipping GPU tests on CI")
            
        self.device = 'cpu'  # Use CPU for testing
        self.detector = YOLODetector(
            model_name='yolov5s',
            conf_threshold=0.45,
            iou_threshold=0.45,
            device=self.device
        )
        
        # Create a simple test image (black with white rectangle)
        self.test_image = np.zeros((416, 416, 3), dtype=np.uint8)
        cv2.rectangle(self.test_image, (100, 100), (300, 300), (255, 255, 255), -1)
        
    def test_detector_initialization(self):
        """Test YOLODetector initialization."""
        self.assertEqual(self.detector.conf_threshold, 0.45)
        self.assertEqual(self.detector.iou_threshold, 0.45)
        self.assertEqual(self.detector.device, self.device)
        self.assertIsNone(self.detector.classes)
        
    def test_detect_empty_image(self):
        """Test detection on empty image."""
        # Create black image
        black_image = np.zeros((416, 416, 3), dtype=np.uint8)
        
        # Should not detect anything in a black image
        detections = self.detector.detect(black_image)
        self.assertEqual(len(detections), 0)
        

if __name__ == '__main__':
    unittest.main()
