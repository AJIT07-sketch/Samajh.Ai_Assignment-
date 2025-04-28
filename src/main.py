import argparse
import time
import cv2
import torch
import os
from pathlib import Path

from detector.detection_model import YOLODetector
from tracker.object_tracker import SimpleTracker
from memory.object_memory import ObjectMemory
from project_utils.visualization import Visualizer
from project_utils.video_utils import VideoWriter


def parse_args():
    parser = argparse.ArgumentParser(description='Real-time detection of missing and new objects')
    parser.add_argument('--input', type=str, required=True, help='Path to input video file or webcam index')
    parser.add_argument('--output', type=str, default='output.mp4', help='Path to output video file')
    parser.add_argument('--model', type=str, default='yolov5s', help='Model to use for detection')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='Confidence threshold for detection')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IoU threshold for NMS')
    parser.add_argument('--device', type=str, default='', help='Device to run inference on (cuda device or cpu)')
    parser.add_argument('--memory-frames', type=int, default=30, help='Number of frames to keep in memory')
    parser.add_argument('--show', action='store_true', help='Show the output in a window')
    parser.add_argument('--classes', nargs='+', type=int, help='Filter by class')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Determine device
    device = args.device if args.device else ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize detector, tracker, memory, and visualizer
    detector = YOLODetector(model_name=args.model, 
                           conf_threshold=args.conf_thres, 
                           iou_threshold=args.iou_thres,
                           device=device, 
                           classes=args.classes)
    
    tracker = SimpleTracker(iou_threshold=0.3)
    object_memory = ObjectMemory(memory_frames=args.memory_frames, 
                                confidence_threshold=0.5,
                                persistence_threshold=0.6)
    visualizer = Visualizer()
    
    # Open video source
    if args.input.isdigit():
        cap = cv2.VideoCapture(int(args.input))
    else:
        cap = cv2.VideoCapture(args.input)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {args.input}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer
    video_writer = None
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        video_writer = VideoWriter(
            output_path.as_posix(),
            fps,
            (width, height)
        )
    
    # Performance metrics
    frame_count = 0
    start_time = time.time()
    processing_times = []
    
    print("Starting video processing...")
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_start_time = time.time()
            
            # Run detection
            detections = detector.detect(frame)
            
            # Update tracker
            tracks = tracker.update(detections)
            
            # Update memory
            missing_objects, new_objects = object_memory.update(tracks)
            
            # Visualize results
            output_frame = visualizer.draw_results(
                frame.copy(), 
                tracks, 
                missing_objects, 
                new_objects
            )
            
            # Calculate processing time for this frame
            frame_time = time.time() - frame_start_time
            processing_times.append(frame_time)
            
            # Display FPS on frame
            current_fps = 1.0 / frame_time if frame_time > 0 else 0
            cv2.putText(output_frame, f"FPS: {current_fps:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display output
            if args.show:
                cv2.imshow('Output', output_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Write to video
            if video_writer:
                video_writer.write(output_frame)
            
            frame_count += 1
            
    except KeyboardInterrupt:
        print("Processing interrupted by user")
    finally:
        # Clean up
        if cap:
            cap.release()
        if video_writer:
            video_writer.release()
        if args.show:
            cv2.destroyAllWindows()
    
    # Print performance statistics
    elapsed_time = time.time() - start_time
    avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
    
    print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Average processing time per frame: {avg_processing_time*1000:.2f} ms")
    

if __name__ == "__main__":
    main()
