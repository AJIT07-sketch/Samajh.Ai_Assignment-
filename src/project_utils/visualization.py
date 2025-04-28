import cv2
import numpy as np
import random


class Visualizer:
    """Class for visualization of detection results."""
    
    def __init__(self):
        """Initialize visualizer."""
        self.colors = {}  # track_id -> color
        self.base_font_scale = 0.5
        self.base_thickness = 2
        
    def draw_results(self, frame, tracks, missing_objects, new_objects):
        """
        Draw detection results on frame.
        
        Args:
            frame: Input frame
            tracks: List of Track objects
            missing_objects: List of missing MemoryObject objects
            new_objects: List of new MemoryObject objects
            
        Returns:
            Frame with visualizations
        """
        height, width = frame.shape[:2]
        
        # Draw tracks
        for track in tracks:
            self._draw_track(frame, track)
            
        # Draw status box
        status_height = 100
        status_width = 300
        status_x = width - status_width - 10
        status_y = 10
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (status_x, status_y), 
                     (status_x + status_width, status_y + status_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw status info
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = status_y + 25
        
        cv2.putText(frame, f"Total objects: {len(tracks)}", 
                   (status_x + 10, y_offset), font, 0.6, (255, 255, 255), 1)
        y_offset += 20
        
        cv2.putText(frame, f"Missing objects: {len(missing_objects)}", 
                   (status_x + 10, y_offset), font, 0.6, (0, 0, 255), 1)
        y_offset += 20
        
        cv2.putText(frame, f"New objects: {len(new_objects)}", 
                   (status_x + 10, y_offset), font, 0.6, (0, 255, 0), 1)
        
        # Draw missing objects indicators
        self._draw_missing_objects(frame, missing_objects)
        
        # Draw new objects indicators
        self._draw_new_objects(frame, new_objects)
        
        return frame
    
    def _draw_track(self, frame, track):
        """Draw track on frame."""
        # Generate color for track if not exists
        if track.track_id not in self.colors:
            self.colors[track.track_id] = self._generate_color()
            
        color = self.colors[track.track_id]
        
        # Draw bounding box
        x1, y1, x2, y2 = [int(v) for v in track.bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.base_thickness)
        
        # Draw track ID and class
        label = f"{track.track_id}: {track.class_name} ({track.confidence:.2f})"
        
        # Draw background for text
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, self.base_font_scale, 1)
        cv2.rectangle(frame, (x1, y1 - text_height - 4), 
                     (x1 + text_width, y1), color, -1)
        
        # Draw text
        cv2.putText(frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, self.base_font_scale, 
                   (255, 255, 255), 1)
        
        # Draw trajectory
        if len(track.trajectory) > 1:
            points = np.array(track.trajectory, dtype=np.int32)
            cv2.polylines(frame, [points], False, color, 2)
        
    def _draw_missing_objects(self, frame, missing_objects):
        """Draw indicators for missing objects."""
        for i, obj in enumerate(missing_objects):
            # Draw last known position
            if obj.track_id in self.colors:
                color = self.colors[obj.track_id]
            else:
                color = (0, 0, 255)  # Red for missing objects
                
            x1, y1, x2, y2 = [int(v) for v in obj.last_bbox]
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Draw circle at last known position
            cv2.circle(frame, (center_x, center_y), 10, color, 2)
            cv2.circle(frame, (center_x, center_y), 15, color, 2)
            
            # Draw X to indicate missing
            cv2.line(frame, (center_x - 10, center_y - 10), 
                    (center_x + 10, center_y + 10), color, 2)
            cv2.line(frame, (center_x + 10, center_y - 10), 
                    (center_x - 10, center_y + 10), color, 2)
            
            # Draw label
            label = f"Missing: {obj.class_name} (ID: {obj.track_id})"
            cv2.putText(frame, label, (center_x - 10, center_y - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, self.base_font_scale, 
                       color, 2)
    
    def _draw_new_objects(self, frame, new_objects):
        """Draw indicators for new objects."""
        for i, obj in enumerate(new_objects):
            # Draw last known position
            if obj.track_id in self.colors:
                color = self.colors[obj.track_id]
            else:
                color = (0, 255, 0)  # Green for new objects
                
            x1, y1, x2, y2 = [int(v) for v in obj.last_bbox]
            
            # Draw animated rectangle (thicker)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw "NEW" label
            label = f"NEW: {obj.class_name}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, self.base_font_scale, 
                       color, 2)
    
    def _generate_color(self):
        """Generate random color for track."""
        return (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
