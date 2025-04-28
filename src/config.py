"""
Configuration parameters for the object detection and tracking system.
"""

# Detection parameters
DETECTION_CONF_THRESHOLD = 0.45  # Confidence threshold for detection
DETECTION_IOU_THRESHOLD = 0.45   # IoU threshold for NMS
DETECTION_CLASSES = None         # None means all classes, or specify list of class ids

# Tracking parameters
TRACKING_IOU_THRESHOLD = 0.3     # IoU threshold for tracking
TRACKING_MAX_AGE = 30            # Maximum age of tracks

# Memory parameters
MEMORY_FRAMES = 30               # Number of frames to keep in memory
MEMORY_CONF_THRESHOLD = 0.5      # Minimum confidence for considering an object
MEMORY_PERSISTENCE_THRESHOLD = 0.6  # Threshold for considering an object persistent

# Visualization parameters
VIZ_SHOW_TRAJECTORIES = True     # Whether to show object trajectories
VIZ_SHOW_IDS = True              # Whether to show object IDs
VIZ_SHOW_CONFIDENCE = True       # Whether to show object confidence

# Video processing parameters
VIDEO_QUEUE_SIZE = 128           # Size of video writer queue
