# Example: Download and prepare ONNX models

## Option 1: Using YOLOv8 (Recommended)

### Install Ultralytics
```bash
pip install ultralytics
```

### Export YOLOv8n for person detection
```python
from ultralytics import YOLO

# Load pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Export to ONNX format
model.export(format='onnx', simplify=True, opset=12)

# Move to models folder
# mv yolov8n.onnx models/person_detection.onnx
```

### Export YOLOv8 for face detection
```python
# You can use a face-specific YOLO model or train your own
# Or use a pre-trained face detection model from:
# https://github.com/ultralytics/yolov5-face

# For now, you can download pre-trained face models from:
# https://huggingface.co/models?search=face+detection+onnx
```

## Option 2: Use pre-trained ONNX models

Download from these sources:

### Person Detection Models:
- YOLOv8n: https://github.com/ultralytics/ultralytics
- YOLOv5: https://github.com/ultralytics/yolov5
- MobileNet-SSD: https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API

### Face Detection Models:
- RetinaFace: https://github.com/onnx/models/tree/main/vision/body_analysis/retinaface
- SCRFD: https://github.com/deepinsight/insightface/tree/master/detection/scrfd
- YuNet: https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet

## Option 3: Use fallback detectors (No ONNX needed)

If you don't have ONNX models, the system automatically uses:
- **HOG + SVM** for person detection (built into OpenCV)
- **Haar Cascade** for face detection (built into OpenCV)

These work out-of-the-box but are less accurate than deep learning models.

## Example: Quick start without models

```bash
# Just run with fallback detectors
python main.py --input input/your_image.jpg --pipeline sequential

# The system will use HOG and Haar Cascade automatically
```
