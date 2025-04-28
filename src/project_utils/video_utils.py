import cv2
import threading
import queue
import time


class VideoWriter:
    """Threaded video writer for better performance."""
    
    def __init__(self, output_path, fps, frame_size, queue_size=128):
        """
        Initialize video writer.
        
        Args:
            output_path: Path to output video file
            fps: Frames per second
            frame_size: Frame size (width, height)
            queue_size: Size of the frame queue
        """
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.queue = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.writer = None
        
        # Start writer thread
        self.thread = threading.Thread(target=self._writer_thread, daemon=True)
        self.thread.start()
        
    def _writer_thread(self):
        """Writer thread function."""
        # Initialize writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            self.output_path, 
            fourcc, 
            self.fps, 
            self.frame_size
        )
        
        while not self.stop_event.is_set():
            try:
                # Get frame with timeout
                frame = self.queue.get(timeout=1.0)
                if frame is None:
                    break
                    
                # Write frame
                self.writer.write(frame)
                self.queue.task_done()
            except queue.Empty:
                continue
                
        # Release writer
        if self.writer:
            self.writer.release()
            
    def write(self, frame):
        """
        Write frame to video.
        
        Args:
            frame: Frame to write
        """
        try:
            self.queue.put(frame, block=False)
        except queue.Full:
            # Skip frame if queue is full
            pass
            
    def release(self):
        """Release resources."""
        # Signal thread to stop
        self.stop_event.set()
        
        # Wait for queue to empty
        remaining = self.queue.qsize()
        if remaining > 0:
            print(f"Waiting for {remaining} frames to be written...")
            
        # Wait for thread to finish
        self.thread.join(timeout=10.0)
        
        if self.thread.is_alive():
            print("Warning: Video writer thread did not terminate properly")
