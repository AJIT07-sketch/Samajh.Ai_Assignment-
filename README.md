 # Real-Time Detection of Object Missing and New Object Placement

This project implements a real-time video analytics pipeline that can detect when objects go missing from a scene and when new objects appear in a scene.

## Features

- **Missing Object Detection**: Identifies when previously present objects are no longer visible
- **New Object Placement Detection**: Detects when new objects appear in the scene
- **Real-time Performance**: Optimized for high-performance real-time processing
- **Multi-threaded Processing**: Uses threading for video writing to maximize FPS
- **Object Memory System**: Tracks objects over time to determine their significance
- **Visualization**: Visual indicators for missing and new objects

## Architecture

The system is built with a modular architecture:

1. **Detection Module**: Uses YOLOv5 for fast and accurate object detection
2. **Tracking Module**: Simple IoU-based tracker with motion prediction
3. **Memory Module**: Maintains object history to determine missing/new objects
4. **Visualization Module**: Provides clear visual feedback on detection results
5. **Video Processing Module**: Optimized for real-time performance

## Requirements

- Python 3.8+
- PyTorch 1.9+
- OpenCV 4.5+
- NumPy
- SciPy
- Other dependencies in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/object-detection-changes.git
cd object-detection-changes
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Build and run with Docker:
```bash
docker build -t object-detection-changes .
docker run --gpus all -v /path/to/videos:/videos object-detection-changes --input /videos/input.mp4 --output /videos/output.mp4
```

## Usage

```bash
python src/main.py --input /path/to/video.mp4 --output output.mp4 --show --model yolov5s.pt
```

### Command Line Arguments

- `--input`: Path to input video file or camera index (e.g., 0 for webcam)
- `--output`: Path to output video file (default: output.mp4)
- `--model`: Model to use for detection (default: yolov5s)
- `--conf-thres`: Confidence threshold for detection (default: 0.45)
- `--iou-thres`: IoU threshold for NMS (default: 0.45)
- `--device`: Device to run inference on (default: cuda if available, otherwise cpu)
- `--memory-frames`: Number of frames to keep in memory (default: 30)
- `--show`: Flag to show output in a window
- `--classes`: Optional filter for specific object classes

## Implementation Details

### Object Detection
The system uses YOLOv5 via PyTorch Hub for object detection. The detection module is designed to be easily replaced with other detection models if needed.

### Object Tracking
A simple IoU (Intersection over Union) based tracker is implemented to track objects across frames. The Hungarian algorithm is used for optimal assignment of detections to existing tracks.

### Memory System
The system maintains a memory of objects that have been detected in the scene. Objects are considered significant based on their persistence and size. The memory system is responsible for determining when an object is missing or new.

### Performance Optimizations
- Threaded video writing to avoid blocking the main processing loop
- Efficient use of PyTorch for inference
- Optimized IoU calculation for tracking

## Performance

The system achieves real-time performance on modern hardware. Actual FPS will depend on:
- Hardware specifications (CPU/GPU)
- Input video resolution
- Model complexity
- Number of objects in the scene

## output 

Sample video 
<video controls src="sample copy.mp4" title="Title"></video>

sample video output 
![alt text](<Screenshot 2025-04-28 215452.png>)
![alt text](<Screenshot 2025-04-28 215500.png>)

sample video 2
<video controls src="sample2.mp4" title="Title"></video>

sample video 2 output 
![alt text](<Screenshot 2025-04-28 222946.png>)
![alt text](<Screenshot 2025-04-28 222959.png>)

sample video  4 
<video controls src="sample4.mp4" title="Title"></video>

sample video 4 output 
![alt text](<Screenshot 2025-04-28 223439.png>)
![alt text](<Screenshot 2025-04-28 223545.png>)


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv5 by Ultralytics
- PyTorch
- OpenCV
