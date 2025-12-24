#!/usr/bin/env python3
"""
Script to download and setup models for human face detection.
Automatically creates folder structure and downloads required ONNX models.
"""

import os
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List

# Base models directory
MODELS_DIR = Path(__file__).parent / "models"

# Model definitions: {folder_name: [(url, filename), ...]}
MODELS_TO_DOWNLOAD: Dict[str, List[tuple]] = {
    # YOLOv8 Face Detection Models
    "YOLOv8-Face": [
        (
            "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.onnx",
            "yolov8n-face.onnx"
        ),
    ],
    
    # YuNet - OpenCV Face Detection (Very Fast)
    "YuNet": [
        (
            "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
            "face_detection_yunet_2023mar.onnx"
        ),
        (
            "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar_int8.onnx",
            "face_detection_yunet_2023mar_int8.onnx"
        ),
    ],
    
    # SCRFD - InsightFace (High Accuracy)  
    "SCRFD": [
        (
            "https://huggingface.co/onnx-community/SCRFD/resolve/main/scrfd_500m_bnkps.onnx",
            "scrfd_500m_bnkps.onnx"
        ),
        (
            "https://huggingface.co/onnx-community/SCRFD/resolve/main/scrfd_2.5g_bnkps.onnx",
            "scrfd_2.5g_bnkps.onnx"
        ),
    ],
    
    # NanoDet - Lightweight Object Detection
    "NanoDet": [
        (
            "https://github.com/opencv/opencv_zoo/raw/main/models/object_detection_nanodet/object_detection_nanodet_2022nov.onnx",
            "object_detection_nanodet_2022nov.onnx"
        ),
        (
            "https://github.com/opencv/opencv_zoo/raw/main/models/object_detection_nanodet/object_detection_nanodet_2022nov_int8.onnx",
            "object_detection_nanodet_2022nov_int8.onnx"
        ),
    ],
    
    # NanoDet-Plus - Enhanced version
    "NanoDet-Plus": [
        (
            "https://github.com/RangiLyu/nanodet/releases/download/v1.0.0-alpha-1/nanodet-plus-m_320.onnx",
            "nanodet-plus-m_320.onnx"
        ),
        (
            "https://github.com/RangiLyu/nanodet/releases/download/v1.0.0-alpha-1/nanodet-plus-m_416.onnx",
            "nanodet-plus-m_416.onnx"
        ),
    ],
    
    # PP-PicoDet - PaddleDetection Lightweight Detector
    "PP-PicoDet": [
        (
            "https://paddledet.bj.bcebos.com/deploy/Inference/picodet_s_320_coco_lcnet.tar",
            "picodet_s_320_coco.tar"
        ),
        (
            "https://paddledet.bj.bcebos.com/deploy/Inference/picodet_m_416_coco_lcnet.tar",
            "picodet_m_416_coco.tar"
        ),
    ],
    
    # EfficientDet-Lite - Lightweight Efficient Detection
    "EfficientDet-Lite": [
        (
            "https://tfhub.dev/tensorflow/lite-model/efficientdet/lite0/detection/metadata/1?lite-format=tflite",
            "efficientdet_lite0.tflite"
        ),
    ],
    
    "EfficientDet-Lite1": [
        (
            "https://tfhub.dev/tensorflow/lite-model/efficientdet/lite1/detection/metadata/1?lite-format=tflite",
            "efficientdet_lite1.tflite"
        ),
    ],
    
    "EfficientDet-Lite2": [
        (
            "https://tfhub.dev/tensorflow/lite-model/efficientdet/lite2/detection/metadata/1?lite-format=tflite",
            "efficientdet_lite2.tflite"
        ),
    ],
    
    # EdgeYOLO - Edge-optimized YOLO
    "EdgeYOLO": [
        (
            "https://huggingface.co/Seeed/edgeyolo/resolve/main/edgeyolo_coco.onnx",
            "edgeyolo_coco.onnx"
        ),
        (
            "https://huggingface.co/Seeed/edgeyolo/resolve/main/edgeyolo_tiny_coco.onnx",
            "edgeyolo_tiny_coco.onnx"
        ),
    ],
    
    # RF-DETR-Nano - Rectified Flow DETR
    "RF-DETR-Nano": [
        (
            "https://huggingface.co/onnx-community/rfdetr_nano-ONNX/resolve/main/onnx/model.onnx",
            "model.onnx"
        ),
        (
            "https://huggingface.co/onnx-community/rfdetr_nano-ONNX/resolve/main/onnx/model_quantized.onnx",
            "model_quantized.onnx"
        ),
        (
            "https://huggingface.co/onnx-community/rfdetr_nano-ONNX/resolve/main/onnx/model_fp16.onnx",
            "model_fp16.onnx"
        ),
        (
            "https://huggingface.co/onnx-community/rfdetr_nano-ONNX/resolve/main/onnx/model_int8.onnx",
            "model_int8.onnx"
        ),
    ],
    
    # YOLOv8 Person Detection
    "YOLOv8-Person": [
        # Note: This needs to be exported from ultralytics
        # Instructions will be provided
    ],
    
    # MediaPipe Face Detection
    "MediaPipe-Face": [
        (
            "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
            "blaze_face_short_range.tflite"
        ),
    ],
    
    # UltraFace - Lightweight Face Detection
    "UltraFace": [
        (
            "https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/raw/master/models/onnx/version-RFB-320.onnx",
            "ultraface_rfb_320.onnx"
        ),
        (
            "https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/raw/master/models/onnx/version-RFB-640.onnx",
            "ultraface_rfb_640.onnx"
        ),
    ],
}


def download_file(url: str, destination: Path, show_progress: bool = True):
    """Download a file with progress bar."""
    try:
        print(f"  Downloading: {url}")
        
        def reporthook(count, block_size, total_size):
            if show_progress and total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                sys.stdout.write(f"\r  Progress: {percent}%")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, destination, reporthook)
        
        if show_progress:
            print()  # New line after progress
            
        file_size_mb = destination.stat().st_size / (1024 * 1024)
        print(f"  ✓ Downloaded: {destination.name} ({file_size_mb:.2f} MB)")
        return True
        
    except Exception as e:
        print(f"  ✗ Error downloading {url}: {e}")
        if destination.exists():
            destination.unlink()
        return False


def create_model_folders():
    """Create necessary folder structure."""
    print(f"Creating models directory structure in: {MODELS_DIR}")
    MODELS_DIR.mkdir(exist_ok=True)
    
    for folder_name in MODELS_TO_DOWNLOAD.keys():
        folder_path = MODELS_DIR / folder_name
        folder_path.mkdir(exist_ok=True)
        print(f"  ✓ Created: {folder_path}")


def download_models(force_redownload: bool = False):
    """Download all models."""
    print("\n" + "="*60)
    print("Starting model downloads...")
    print("="*60)
    
    total_models = sum(len(files) for files in MODELS_TO_DOWNLOAD.values())
    downloaded = 0
    skipped = 0
    failed = 0
    
    for folder_name, files in MODELS_TO_DOWNLOAD.items():
        if not files:
            print(f"\n[{folder_name}] - Manual setup required")
            continue
            
        print(f"\n[{folder_name}]")
        folder_path = MODELS_DIR / folder_name
        
        for url, filename in files:
            destination = folder_path / filename
            
            # Skip if file exists and not forcing redownload
            if destination.exists() and not force_redownload:
                file_size_mb = destination.stat().st_size / (1024 * 1024)
                print(f"  ⊙ Skipped (exists): {filename} ({file_size_mb:.2f} MB)")
                skipped += 1
                continue
            
            # Download the file
            if download_file(url, destination):
                downloaded += 1
            else:
                failed += 1
    
    # Summary
    print("\n" + "="*60)
    print("Download Summary:")
    print("="*60)
    print(f"Total models: {total_models}")
    print(f"Downloaded:   {downloaded}")
    print(f"Skipped:      {skipped}")
    print(f"Failed:       {failed}")
    print("="*60)


def create_readme():
    """Create a README with model information."""
    readme_path = MODELS_DIR / "DOWNLOADED_MODELS.md"
    
    content = """# Downloaded Models

This folder contains ONNX and TFLite models for face and person detection.

## Models Downloaded by Script

### Face Detection Models

#### YOLOv8-Face
- **File**: `yolov8n-face.onnx`
- **Purpose**: Fast face detection using YOLOv8 architecture
- **Input Size**: 640x640
- **Use Case**: General purpose face detection

#### YuNet (OpenCV)
- **Files**: 
  - `face_detection_yunet_2023mar.onnx` (FP32)
  - `face_detection_yunet_2023mar_int8.onnx` (INT8 quantized)
- **Purpose**: Lightweight and fast face detection
- **Input Size**: 160x120 or 320x320
- **Use Case**: Real-time applications, embedded systems

#### SCRFD (InsightFace)
- **Files**:
  - `scrfd_500m_bnkps.onnx` (500M params)
  - `scrfd_2.5g_bnkps.onnx` (2.5G params)
- **Purpose**: High-accuracy face detection with keypoints
- **Input Size**: Variable
- **Use Case**: High-quality face detection, face alignment

#### UltraFace
- **Files**:
  - `ultraface_rfb_320.onnx` (320x240 input)
  - `ultraface_rfb_640.onnx` (640x480 input)
- **Purpose**: Ultra-lightweight face detection (~1MB model)
- **Input Size**: 320x240 or 640x480
- **Use Case**: Edge devices, mobile applications

#### MediaPipe Face
- **File**: `blaze_face_short_range.tflite`
- **Purpose**: Google's BlazeFace for short-range detection
- **Format**: TFLite (needs conversion for ONNX)
- **Use Case**: Mobile and web applications

### Object Detection Models (for Person Detection)

#### NanoDet
- **Files**:
  - `object_detection_nanodet_2022nov.onnx` (FP32)
  - `object_detection_nanodet_2022nov_int8.onnx` (INT8)
- **Purpose**: FCOS-style anchor-free object detection
- **Input Size**: 320x320 or 416x416
- **Use Case**: Lightweight real-time detection
- **Classes**: 80 COCO classes including person

#### NanoDet-Plus
- **Files**:
  - `nanodet-plus-m_320.onnx` (320 input)
  - `nanodet-plus-m_416.onnx` (416 input)
- **Purpose**: Enhanced NanoDet with better accuracy
- **Features**: AGM + DSLA for optimal label assignment
- **Input Size**: 320x320 or 416x416
- **Use Case**: Improved detection on lightweight devices

#### PP-PicoDet (PaddleDetection)
- **Files**:
  - `picodet_s_416_coco_lcnet.onnx` (Small)
  - `picodet_m_416_coco_lcnet.onnx` (Medium)
- **Purpose**: Ultra-lightweight detector from Baidu
- **Input Size**: 416x416
- **Use Case**: Mobile and edge deployment
- **Classes**: 80 COCO classes

#### EfficientDet-Lite
- **Files**:
  - `efficientdet_lite0.tflite` (Lite0 - smallest)
  - `efficientdet_lite1.tflite` (Lite1 - balanced)
  - `efficientdet_lite2.tflite` (Lite2 - more accurate)
- **Purpose**: Efficient scalable detection
- **Format**: TFLite
- **Use Case**: Mobile devices, edge TPU
- **Classes**: 90 COCO classes

#### EdgeYOLO
- **Files**:
  - `edgeyolo_coco.onnx` (Standard)
  - `edgeyolo_tiny_coco.onnx` (Tiny version)
- **Purpose**: Edge-optimized YOLO variant
- **Input Size**: 640x640
- **Use Case**: Edge devices with limited resources
- **Classes**: 80 COCO classes

#### RF-DETR-Nano
- **Files**:
  - `model.onnx` (FP32)
  - `model_fp16.onnx` (FP16)
  - `model_int8.onnx` (INT8)
  - `model_quantized.onnx` (Dynamic quantized)
- **Purpose**: Rectified Flow DETR - transformer-based detection
- **Input Size**: Variable
- **Use Case**: High accuracy detection, research
- **Classes**: 80 COCO classes

## Manual Setup Required

### YOLOv8-Person
For person detection, you need to export YOLOv8:

```bash
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx')"
mv yolov8n.onnx models/YOLOv8-Person/
```

## Usage Examples

### Face Detection (OpenCV + YuNet)
```python
import cv2

detector = cv2.FaceDetectorYN.create(
    "models/YuNet/face_detection_yunet_2023mar.onnx",
    "", 
    (320, 320)
)

img = cv2.imread("image.jpg")
faces = detector.detect(img)
```

### Object Detection (NanoDet)
```python
import cv2
import numpy as np

net = cv2.dnn.readNet("models/NanoDet/object_detection_nanodet_2022nov.onnx")
blob = cv2.dnn.blobFromImage(img, 1.0/255.0, (320, 320), [0,0,0], swapRB=True, crop=False)
net.setInput(blob)
outputs = net.forward(net.getUnconnectedOutLayersNames())
```

### ONNX Runtime
```python
import onnxruntime as ort

session = ort.InferenceSession("models/RF-DETR-Nano/model.onnx")
outputs = session.run(None, {input_name: input_data})
```

## Model Selection Guide

### For Face Detection:
- **Fastest**: YuNet INT8 (~1ms on CPU)
- **Balanced**: UltraFace 320, YOLOv8-Face
- **Most Accurate**: SCRFD 2.5G
- **Smallest Size**: UltraFace (~1MB)
- **Best for Mobile**: MediaPipe BlazeFace

### For Person Detection:
- **Fastest**: NanoDet INT8, PP-PicoDet-S
- **Balanced**: NanoDet-Plus, EdgeYOLO Tiny
- **Most Accurate**: RF-DETR-Nano FP32
- **Smallest Size**: EfficientDet-Lite0
- **Best for Edge**: PP-PicoDet, EdgeYOLO

### By Input Size:
- **320x320**: NanoDet, NanoDet-Plus, YuNet
- **416x416**: NanoDet-Plus, PP-PicoDet
- **640x640**: YOLOv8, EdgeYOLO

## Performance Tips

1. **Use INT8 quantized models** for faster inference on CPU
2. **Choose smaller input sizes** (320x320) for real-time applications
3. **Use FP16 models** on GPUs that support half-precision
4. **Batch processing** when processing multiple images
5. **Model warmup** - run inference once before timing

## COCO Classes (Person = class 0)

All object detection models are trained on COCO dataset:
- Class 0: person
- 79 other classes (car, bicycle, etc.)

Filter results to only detect persons by checking class_id == 0.

"""
    
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"\n✓ Created model documentation: {readme_path}")


def print_usage():
    """Print usage instructions."""
    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print("\n1. Check downloaded models in: models/")
    print("\n2. For YOLOv8 Person Detection, run:")
    print("   pip install ultralytics")
    print("   python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx')\"")
    print("\n3. Update your config.py to use downloaded models")
    print("\n4. Test with: python main.py")
    print("="*60)


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download face and person detection models"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist"
    )
    parser.add_argument(
        "--no-readme",
        action="store_true",
        help="Skip creating README file"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Model Download Script for Face Detection")
    print("="*60)
    
    # Create folder structure
    create_model_folders()
    
    # Download models
    download_models(force_redownload=args.force)
    
    # Create documentation
    if not args.no_readme:
        create_readme()
    
    # Print next steps
    print_usage()
    
    print("\n✓ Setup complete!")


if __name__ == "__main__":
    main()
