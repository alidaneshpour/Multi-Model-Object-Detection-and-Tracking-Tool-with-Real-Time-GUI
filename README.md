# Multi-Model Object Detection and Tracking Tool with Real-Time GUI

A comprehensive computer vision application that integrates multiple state-of-the-art object detection models with advanced tracking capabilities, featuring an intuitive real-time GUI for video analysis and monitoring applications.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🚀 Features

### Multi-Model Detection Framework
- **YOLOv5 Integration**: Ultra-fast real-time object detection with high accuracy
- **YOLOv8 Support**: Latest YOLO architecture with improved performance metrics  
- **Mask R-CNN Implementation**: Instance segmentation with precise bounding box detection
- **Dynamic Model Switching**: Seamless comparison between different detection approaches

### Advanced Tracking System
- **CSRT Tracker Integration**: Discriminative Correlation Filter with Channel and Spatial Reliability
- **Single Object Tracking**: High-precision tracking of user-selected objects
- **Multi-Object Tracking**: Simultaneous tracking of multiple detected objects
- **Real-Time Performance**: Optimized for live video processing

### Intelligent Control Features
- **Dynamic Confidence Thresholding**: Real-time adjustment of detection sensitivity (0-100%)
- **Interactive Object Selection**: User-friendly object ID selection with confidence scoring
- **Confidence Score Visualization**: Optional display of detection confidence levels
- **Automatic Object Filtering**: Smart filtering based on user-defined confidence thresholds

### User Interface
- **Intuitive GUI**: Clean Tkinter-based interface with three-panel video display
- **Real-Time Visualization**: Live video processing with immediate visual feedback
- **Threaded Processing**: Non-blocking video playback and processing
- **Fullscreen Support**: Press Escape to toggle fullscreen mode

## 🏗️ Architecture

<p align="center">
  <img src="readme_images/GUI_Environment.PNG" alt="GUI Environment" width="900">
</p>

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.8+**: Primary programming language
- **OpenCV 4.5+**: Computer vision and video processing
- **PyTorch**: Deep learning framework for model inference
- **Torchvision**: Pre-trained models and transforms
- **Ultralytics**: YOLOv5 and YOLOv8 implementations

### Detection Models
- **YOLOv5s**: `torch.hub.load('ultralytics/yolov5', 'yolov5s')`
- **YOLOv8s**: `YOLO('yolov8s.pt')` via Ultralytics
- **Mask R-CNN**: `torchvision.models.detection.maskrcnn_resnet50_fpn`

### GUI Framework
- **Tkinter**: Native Python GUI toolkit
- **PIL (Pillow)**: Image processing and display
- **Threading**: Non-blocking video processing

## 📋 Prerequisites

```bash
# Core dependencies
opencv-python>=4.5.0
torch>=1.9.0
torchvision>=0.10.0
ultralytics>=8.0.0
Pillow>=8.0.0

# Additional requirements
numpy>=1.21.0
matplotlib>=3.3.0
```

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/multi-model-object-detection-tracking.git
cd multi-model-object-detection-tracking
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download YOLOv8 weights** (automatic on first run)
```bash
# YOLOv8 weights will be downloaded automatically
# YOLOv5 weights are loaded from torch.hub
# Mask R-CNN uses pre-trained torchvision weights
```

## 💻 Usage

### Quick Start

1. **Launch the application**
```bash
python main.py
```

2. **Upload a video file**
   - Click "Upload Video" button
   - Select video file (.mp4, .avi, .mov)
   - Video appears in the left panel

3. **Run object detection**
   - Choose detection model: YOLOv5, YOLOv8, or Mask R-CNN
   - Adjust confidence threshold (default: 70%)
   - Detection results appear in the center panel

4. **Track objects**
   - Single object: Select object ID from detection results
   - Multiple objects: Enter comma-separated IDs or "all"
   - Tracking results appear in the right panel

### Advanced Configuration

#### Confidence Threshold Control
```python
# Adjust detection sensitivity
confidence_threshold = 0.7  # 70% confidence
# Objects with confidence below threshold are filtered out
```

#### Model Selection Guidelines
- **YOLOv5**: Best balance of speed and accuracy
- **YOLOv8**: Latest architecture with improved performance
- **Mask R-CNN**: Highest accuracy with instance segmentation

#### Tracking Parameters
```python
# CSRT Tracker Configuration
tracker = cv2.legacy.TrackerCSRT_create()
# Automatically optimized for real-time performance
```

## 🔧 Code Structure

```
project/
├── main.py                 # Main application entry point
├── models/
│   ├── yolo_detector.py   # YOLOv5/v8 detection logic
│   ├── maskrcnn_detector.py # Mask R-CNN implementation
│   └── tracker.py         # Object tracking algorithms
├── gui/
│   ├── main_window.py     # Primary GUI components
│   ├── video_player.py    # Video display functionality
│   └── controls.py        # User control interfaces
├── utils/
│   ├── video_utils.py     # Video processing utilities
│   ├── detection_utils.py # Detection post-processing
│   └── visualization.py   # Bounding box rendering
├── config/
│   ├── settings.py        # Application configuration
│   └── model_config.py    # Model-specific parameters
└── requirements.txt       # Python dependencies
```

## 🎮 Key Features Showcase

### Dynamic Confidence Control
- Real-time threshold adjustment without reprocessing
- Confidence score visualization for each detected object
- Smart filtering based on user preferences

### Multi-Object Tracking Excellence
- Simultaneous tracking of up to 10+ objects
- Robust performance under occlusion and lighting changes
- Automatic track recovery and re-identification

### Professional GUI Design
- Three-panel layout for comprehensive workflow visualization
- Responsive design with fullscreen capabilities
- Intuitive controls with immediate visual feedback

## 🚨 Troubleshooting

### Common Issues

**1. Model Loading Errors**
```bash
# Ensure proper PyTorch installation
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**2. Video Codec Issues**
```bash
# Install additional codecs
pip install opencv-python-headless
```

**3. GUI Display Problems**
```bash
# Update Tkinter (Linux)
sudo apt-get install python3-tk
```

**4. Memory Issues with Large Videos**
- Reduce video resolution before processing
- Increase confidence threshold to reduce detections
- Use YOLOv5 instead of Mask R-CNN for faster processing

## 🔬 Technical Implementation Details

### Detection Pipeline
1. **Frame Extraction**: OpenCV captures video frames
2. **Preprocessing**: Image transformation and normalization
3. **Model Inference**: Forward pass through selected detection model
4. **Post-processing**: NMS, confidence filtering, coordinate transformation
5. **Visualization**: Bounding box rendering with confidence scores

### Tracking Algorithm
```python
# CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability)
# - Robust to appearance changes
# - Handles partial occlusions
# - Real-time performance optimization
tracker = cv2.legacy.TrackerCSRT_create()
```

## 🚀 Future Enhancements

- [ ] **Real-time Webcam Support**: Live camera feed processing
- [ ] **Video Export Functionality**: Save processed videos with annotations
- [ ] **Batch Processing**: Multiple video file processing queue
- [ ] **Custom Model Training**: Train on custom datasets
- [ ] **API Integration**: RESTful API for remote processing
- [ ] **Database Integration**: Store detection results and analytics
- [ ] **Advanced Analytics**: Object counting, trajectory analysis
- [ ] **Mobile App**: Companion mobile application

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black main.py
flake8 main.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics**: YOLOv5 and YOLOv8 implementations
- **Facebook Research**: Mask R-CNN architecture
- **OpenCV Community**: Computer vision algorithms and tools
- **PyTorch Team**: Deep learning framework and pre-trained models

## 📞 Contact

- **Developer**: [Your Name]
- **Email**: [your.email@example.com]
- **LinkedIn**: [Your LinkedIn Profile]
- **GitHub**: [Your GitHub Profile]

---

⭐ **Star this repository** if you find it helpful!

[![GitHub stars](https://img.shields.io/github/stars/yourusername/multi-model-object-detection-tracking.svg?style=social&label=Star&maxAge=2592000)](https://GitHub.com/yourusername/multi-model-object-detection-tracking/stargazers/)
