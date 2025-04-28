class MemoryObject:
    """Class to represent an object in memory."""
    
    def __init__(self, track, memory_frames=30):
        """
        Initialize a memory object from a track.
        
        Args:
            track: Track object
            memory_frames: Number of frames to keep in memory
        """
        self.track_id = track.track_id
        self.class_id = track.class_id
        self.class_name = track.class_name
        self.first_bbox = track.bbox.copy()
        self.last_bbox = track.bbox.copy()
        self.first_centroid = track.centroid
        self.last_centroid = track.centroid
        self.last_confidence = track.confidence
        self.presence_history = [1]  # 1 for present, 0 for absent
        self.memory_frames = memory_frames
        self.consecutive_missing_count = 0
        self.status = "present"  # present, missing, new
        self.is_significant = False  # Whether the object is significant enough to track
        self.significance_score = 0.0  # Score indicating object significance
        self.frame_count = 1  # Number of frames the object has been tracked
        
    def update(self, track=None):
        """
        Update memory object with new track.
        
        Args:
            track: Track object or None if not detected
        """
        if track:
            self.last_bbox = track.bbox.copy()
            self.last_centroid = track.centroid
            self.last_confidence = track.confidence
            self.presence_history.append(1)
            self.consecutive_missing_count = 0
            self.status = "present"
            self.frame_count += 1
        else:
            self.presence_history.append(0)
            self.consecutive_missing_count += 1
            if self.consecutive_missing_count >= 5 and self.is_significant:
                self.status = "missing"
                
        # Keep history within memory limit
        if len(self.presence_history) > self.memory_frames:
            self.presence_history.pop(0)
            
        # Update significance based on consistency of appearance
        presence_rate = sum(self.presence_history) / len(self.presence_history)
        size_score = self._calculate_size_score(self.last_bbox)
        
        # Objects that appear in many frames and have reasonable size are more significant
        self.significance_score = presence_rate * min(1.0, self.frame_count / 10) * size_score
        self.is_significant = self.significance_score > 0.4 and self.frame_count >= 5
        
    def _calculate_size_score(self, bbox):
        """Calculate size score based on bbox size."""
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height
        
        # Normalize area to a score between 0 and 1
        # Assuming a reasonable object is at least 50x50 but not more than half the frame
        # These values might need adjustment based on your specific use case
        min_area = 50 * 50
        max_area = 1920 * 1080 / 2  # Half of a full HD frame
        
        if area < min_area:
            return area / min_area
        elif area > max_area:
            return max(0.5, 1.0 - (area - max_area) / max_area)
        else:
            return 1.0


class ObjectMemory:
    """Class to manage object memory and detect missing/new objects."""
    
    def __init__(self, memory_frames=30, confidence_threshold=0.5, persistence_threshold=0.6):
        """
        Initialize object memory.
        
        Args:
            memory_frames: Number of frames to keep in memory
            confidence_threshold: Minimum confidence for considering an object
            persistence_threshold: Threshold for considering an object persistent
        """
        self.memory_objects = {}  # track_id -> MemoryObject
        self.memory_frames = memory_frames
        self.confidence_threshold = confidence_threshold
        self.persistence_threshold = persistence_threshold
        self.frame_count = 0
        
    def update(self, tracks):
        """
        Update memory with new tracks.
        
        Args:
            tracks: List of Track objects
            
        Returns:
            Tuple of (missing_objects, new_objects)
        """
        self.frame_count += 1
        
        # Convert tracks to dictionary for easier lookup
        track_dict = {track.track_id: track for track in tracks}
        
        # Update existing memory objects
        for track_id in list(self.memory_objects.keys()):
            if track_id in track_dict:
                # Object is still present
                self.memory_objects[track_id].update(track_dict[track_id])
            else:
                # Object is not detected in current frame
                self.memory_objects[track_id].update(None)
                
        # Add new objects
        for track_id, track in track_dict.items():
            if track_id not in self.memory_objects and track.confidence >= self.confidence_threshold:
                self.memory_objects[track_id] = MemoryObject(track, self.memory_frames)
                
        # Identify missing and new objects
        missing_objects = []
        new_objects = []
        
        for track_id, memory_obj in self.memory_objects.items():
            # Missing objects: previously significant objects that are now missing
            if memory_obj.status == "missing":
                missing_objects.append(memory_obj)
                
            # New objects: objects that have been around for a few frames but were just confirmed as significant
            if track_id in track_dict and memory_obj.is_significant and memory_obj.frame_count <= 10:
                new_objects.append(memory_obj)
                
        # Clean up old memory objects that are no longer relevant
        for track_id in list(self.memory_objects.keys()):
            memory_obj = self.memory_objects[track_id]
            if memory_obj.consecutive_missing_count > self.memory_frames:
                del self.memory_objects[track_id]
                
        return missing_objects, new_objects
