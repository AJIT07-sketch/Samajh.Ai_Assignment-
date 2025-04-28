import numpy as np
from scipy.optimize import linear_sum_assignment


class Track:
    def __init__(self, detection, track_id, class_names=None):
        """
        Initialize a new track.

        Args:
            detection: Detection object (dictionary)
            track_id: Unique track ID
            class_names: Optional dictionary mapping class IDs to class names
        """
        self.bbox = detection['box']  # Bounding box [x1, y1, x2, y2]
        self.confidence = detection['confidence']  # Confidence score
        self.class_id = detection['class']  # Class ID
        self.track_id = track_id
        self.age = 0
        self.hits = 0
        self.missed = 0

        # Calculate centroid from bounding box
        x1, y1, x2, y2 = self.bbox
        self.centroid = ((x1 + x2) / 2, (y1 + y2) / 2)  # (center_x, center_y)

        # Set class name if class_names mapping is provided
        self.class_name = class_names[self.class_id] if class_names and self.class_id in class_names else str(self.class_id)

        # Initialize trajectory with the first centroid
        self.trajectory = [self.centroid]

    def update(self, detection):
        """
        Update track with new detection.

        Args:
            detection: Detection object
        """
        self.bbox = detection['box']  # Update to use dictionary key
        self.confidence = detection['confidence']  # Update to use dictionary key
        self.class_id = detection['class']  # Update to use dictionary key
        self.age += 1
        self.hits += 1

        # Update centroid and add it to the trajectory
        x1, y1, x2, y2 = self.bbox
        self.centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
        self.trajectory.append(self.centroid)
        
    def mark_missed(self):
        """Mark the track as missed in current frame."""
        self.missed += 1
        
    def is_confirmed(self):
        """Check if the track is confirmed (seen multiple times)."""
        return self.hits >= 3
        
    def should_be_deleted(self):
        """Check if the track should be deleted."""
        return self.missed >= 30


class SimpleTracker:
    """Simple IoU-based object tracker."""
    
    def __init__(self, iou_threshold=0.3, max_age=30):
        """
        Initialize tracker.
        
        Args:
            iou_threshold: IoU threshold for matching
            max_age: Maximum number of frames to keep invisible tracks
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks = []
        self.next_track_id = 1
        
    def update(self, detections):
        """
        Update tracker with new detections.
        
        Args:
            detections: List of Detection objects (dictionaries)
            
        Returns:
            List of Track objects
        """
        # Handle first frame
        if not self.tracks:
            for detection in detections:
                self._init_track(detection)
            return self.tracks
            
        # Calculate IoU between tracks and new detections
        if self.tracks and detections:
            iou_matrix = np.zeros((len(self.tracks), len(detections)))
            
            for i, track in enumerate(self.tracks):
                for j, detection in enumerate(detections):
                    iou_matrix[i, j] = self._calculate_iou(track.bbox, detection['box'])  # Access bbox using dictionary key
                    
            # Apply Hungarian algorithm for optimal assignment
            track_indices, detection_indices = linear_sum_assignment(-iou_matrix)
            
            # Update matched tracks
            for track_idx, detection_idx in zip(track_indices, detection_indices):
                if iou_matrix[track_idx, detection_idx] >= self.iou_threshold:
                    self.tracks[track_idx].update(detections[detection_idx])
                    detections[detection_idx]['track_id'] = self.tracks[track_idx].track_id
                else:
                    # Mark track as invisible
                    self.tracks[track_idx].mark_missed()
                    
            # Find unmatched detections
            matched_detection_indices = set(detection_indices[iou_matrix[track_indices, detection_indices] >= self.iou_threshold])
            unmatched_detection_indices = [i for i in range(len(detections)) if i not in matched_detection_indices]
            
            # Initialize new tracks for unmatched detections
            for idx in unmatched_detection_indices:
                self._init_track(detections[idx])
                
            # Mark unmatched tracks as invisible
            matched_track_indices = set(track_indices[iou_matrix[track_indices, detection_indices] >= self.iou_threshold])
            for i in range(len(self.tracks)):
                if i not in matched_track_indices:
                    self.tracks[i].mark_missed()
                    
        else:
            # If there are tracks but no detections, mark all tracks as invisible
            if self.tracks:
                for track in self.tracks:
                    track.mark_missed()
                    
            # If there are detections but no tracks, create new tracks
            if detections:
                for detection in detections:
                    self._init_track(detection)
                    
        # Remove old tracks
        self.tracks = [track for track in self.tracks if not track.should_be_deleted()]
        
        return self.tracks
    
    def _init_track(self, detection):
        """Initialize a new track from detection."""
        # Example class names mapping (replace with your actual mapping if available)
        class_names = {0: "person", 1: "bicycle", 2: "car"}  # Example mapping
        track = Track(detection, self.next_track_id, class_names=class_names)
        detection['track_id'] = self.next_track_id  # Store track_id in the dictionary
        self.next_track_id += 1
        self.tracks.append(track)
        
    def _calculate_iou(self, bbox1, bbox2):
        """
        Calculate IoU between two bounding boxes.
        
        Args:
            bbox1: [x1, y1, x2, y2]
            bbox2: [x1, y1, x2, y2]
            
        Returns:
            IoU value
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        # Check if there is an intersection
        if x2 < x1 or y2 < y1:
            return 0.0
            
        intersection_area = (x2 - x1) * (y2 - y1)
        
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        union_area = bbox1_area + bbox2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
