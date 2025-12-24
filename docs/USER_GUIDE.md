# User Guide - Human & Face Detection System

## 📖 Table of Contents
1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Advanced Usage](#advanced-usage)
5. [Model Management](#model-management)
6. [Troubleshooting](#troubleshooting)
7. [Examples](#examples)

---

## Quick Start

### Minimal Example
```bash
# Process an image
python main.py --input input/photo.jpg

# Process a video
python main.py --input input/video.mp4
```

That's it! The system will:
- ✅ Detect all persons
- ✅ Detect all faces
- ✅ Save cropped images to `output/`
- ✅ Generate performance metrics

---

## Installation

### 1. System Requirements

**Hardware:**
- Orange Pi RV 2 (or any Linux SBC)
- 4GB RAM minimum
- ARM or x86 CPU

**Software:**
- Python 3.8+
- OpenCV 4.5+
- ONNX Runtime

### 2. Install Dependencies

```bash
# Clone repository
git clone <repo-url>
cd detected-human-faces

# Install Python packages
pip install -r requirements.txt

# Or using conda
conda env create -f environment.yml
conda activate detection
```

### 3. Download Models

```bash
# Download all available models
python scripts/download_models.py

# Check downloaded models
ls -lh models/*/
```

### 4. Verify Installation

```bash
# Run example script
python example.py

# Or test with a sample image
python main.py --input input/test.png
```

---

## Basic Usage

### Command Line Interface

```bash
python main.py [OPTIONS]
```

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--input`, `-i` | Input image or video file | `--input input/photo.jpg` |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--pipeline`, `-p` | `sequential` | Pipeline mode: `sequential` or `parallel` |
| `--person-model` | `models/person_detection.onnx` | Path to person detection model |
| `--face-model` | `models/face_detection.onnx` | Path to face detection model |
| `--person-threshold` | `0.5` | Confidence threshold for person detection (0-1) |
| `--face-threshold` | `0.5` | Confidence threshold for face detection (0-1) |
| `--output-dir` | `output/` | Directory to save output files |

### Examples

#### 1. Process Image with Default Settings
```bash
python main.py --input input/photo.jpg
```

#### 2. Process Video with Parallel Pipeline
```bash
python main.py --input input/video.mp4 --pipeline parallel
```

#### 3. Use Custom Models
```bash
python main.py \
    --input input/test.jpg \
    --person-model models/NanoDet/object_detection_nanodet_2022nov.onnx \
    --face-model models/YuNet/face_detection_yunet_2023mar.onnx
```

#### 4. Adjust Detection Thresholds
```bash
python main.py \
    --input input/photo.jpg \
    --person-threshold 0.6 \
    --face-threshold 0.7
```

#### 5. Custom Output Directory
```bash
python main.py \
    --input input/photo.jpg \
    --output-dir results/experiment_1/
```

---

## Advanced Usage

### Pipeline Modes

#### Sequential Pipeline (Default)
Best for: Crowded scenes, multiple persons

```bash
python main.py --input input/crowd.jpg --pipeline sequential
```

**How it works:**
1. Detects all persons in the image
2. For each person:
   - Crops the person region
   - Detects faces within that crop
   - Saves person and face crops

**Advantages:**
- Better face-person association
- Faster face detection (smaller search area)
- More accurate in crowded scenes

#### Parallel Pipeline
Best for: Single person, speed critical

```bash
python main.py --input input/portrait.jpg --pipeline parallel
```

**How it works:**
1. Detects persons and faces simultaneously (parallel threads)
2. Merges results
3. Saves all detections

**Advantages:**
- Faster processing
- Good for single-person scenarios
- Independent detections

### Confidence Thresholds

#### Lower Threshold = More Detections
```bash
# Detect more persons (may include false positives)
python main.py --input input/photo.jpg --person-threshold 0.3
```

#### Higher Threshold = Fewer, More Confident Detections
```bash
# Only high-confidence detections
python main.py --input input/photo.jpg --person-threshold 0.7
```

#### Balanced Settings
```bash
# Good balance for most cases
python main.py --input input/photo.jpg \
    --person-threshold 0.5 \
    --face-threshold 0.5
```

### Batch Processing

Process multiple files:

```bash
#!/bin/bash
# process_batch.sh

for img in input/*.jpg; do
    echo "Processing: $img"
    python main.py --input "$img" --output-dir "output/$(basename $img .jpg)/"
done
```

### Video Processing

```bash
# Process video frame by frame
python main.py --input input/video.mp4 --pipeline sequential

# Faster video processing (parallel)
python main.py --input input/video.mp4 --pipeline parallel
```

**Output:**
- Frame-by-frame crops: `output/video_frame_000001_person_0.jpg`
- Performance summary: `logs/video_summary.txt`

---

## Model Management

### Available Models

Check downloaded models:
```bash
find models/ -name "*.onnx" -o -name "*.tflite"
```

### Model Categories

#### Face Detection Models
| Model | Size | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| YuNet | 228KB | ⚡⚡⚡ | ⭐⭐ | Real-time, embedded |
| YuNet INT8 | 99KB | ⚡⚡⚡⚡ | ⭐⭐ | Ultra-fast, low power |
| YOLOv8-Face | 12MB | ⚡⚡ | ⭐⭐⭐ | Balanced, general |
| UltraFace 320 | 1.3MB | ⚡⚡⚡ | ⭐⭐ | Mobile, edge devices |
| UltraFace 640 | 1.6MB | ⚡⚡ | ⭐⭐⭐ | Better accuracy |
| SCRFD | TBD | ⚡ | ⭐⭐⭐⭐ | High accuracy |

#### Object Detection Models (for Person)
| Model | Size | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| NanoDet | 3.6MB | ⚡⚡⚡ | ⭐⭐⭐ | Lightweight, balanced |
| NanoDet INT8 | 1MB | ⚡⚡⚡⚡ | ⭐⭐ | Ultra-fast |
| NanoDet-Plus 320 | 4.6MB | ⚡⚡⚡ | ⭐⭐⭐ | Enhanced NanoDet |
| RF-DETR-Nano | 103MB | ⚡ | ⭐⭐⭐⭐ | High accuracy |
| RF-DETR INT8 | 27MB | ⚡⚡ | ⭐⭐⭐ | Quantized DETR |
| EfficientDet-Lite0 | 4.4MB | ⚡⚡ | ⭐⭐⭐ | Mobile-optimized |

### Using Different Models

#### Example 1: Fastest Configuration
```bash
python main.py \
    --input input/photo.jpg \
    --person-model models/NanoDet/object_detection_nanodet_2022nov_int8.onnx \
    --face-model models/YuNet/face_detection_yunet_2023mar_int8.onnx \
    --pipeline parallel
```

#### Example 2: Most Accurate Configuration
```bash
python main.py \
    --input input/photo.jpg \
    --person-model models/RF-DETR-Nano/model.onnx \
    --face-model models/YOLOv8-Face/yolov8n-face.onnx \
    --pipeline sequential \
    --person-threshold 0.6 \
    --face-threshold 0.6
```

#### Example 3: Balanced Configuration
```bash
python main.py \
    --input input/photo.jpg \
    --person-model models/NanoDet-Plus/nanodet-plus-m_416.onnx \
    --face-model models/UltraFace/ultraface_rfb_640.onnx
```

### Model Testing Script

Test all models automatically:
```bash
bash scripts/test_all_models.sh input/test.jpg
```

---

## Output Files

### Directory Structure

```
output/
├── photo_person_0.jpg          # Cropped person 0
├── photo_person_1.jpg          # Cropped person 1
├── photo_face_0_0.jpg          # Face 0 from person 0
├── photo_face_0_1.jpg          # Face 1 from person 0
├── photo_face_1_0.jpg          # Face 0 from person 1
└── photo_annotated.jpg         # Original with bounding boxes

logs/
└── photo_summary.txt           # Performance metrics
```

### Naming Convention

**Person crops:** `{input_name}_person_{index}.jpg`
- Example: `photo_person_0.jpg`, `photo_person_1.jpg`

**Face crops:** `{input_name}_face_{person_index}_{face_index}.jpg`
- Example: `photo_face_0_0.jpg` (1st face of 1st person)

**Video frames:** `{video_name}_frame_{frame_num}_person_{index}.jpg`
- Example: `video_frame_000123_person_0.jpg`

---

## Performance Metrics

### Viewing Metrics

Metrics are automatically displayed after processing:
```
============================================================
PERFORMANCE REPORT
============================================================
Total frames processed: 1
Total processing time: 2.34s
Average person processing time: 156.78ms

FPS Statistics:
  Max FPS: 4.56
  Min FPS: 4.56
  Avg FPS: 4.56

Accuracy Statistics:
  Max Accuracy: 0.8234
  Min Accuracy: 0.8234
  Avg Accuracy: 0.8234
============================================================
```

### Metrics Log File

Detailed metrics saved to `logs/{input_name}_summary.txt`

### Understanding Metrics

| Metric | Description | Good Range |
|--------|-------------|------------|
| FPS | Frames per second | 2-10 FPS (Orange Pi) |
| Processing Time | Time per frame/image | <500ms (image) |
| Accuracy | Detection confidence | >0.7 |

---

## Troubleshooting

### Common Issues

#### 1. Model Not Found
```
Error: Model not found at models/person_detection.onnx
```

**Solution:**
```bash
# Download models
python scripts/download_models.py

# Or specify existing model
python main.py --input input/photo.jpg \
    --person-model models/NanoDet/object_detection_nanodet_2022nov.onnx
```

#### 2. Out of Memory
```
Error: Cannot allocate memory
```

**Solution:**
```bash
# Use quantized models (smaller)
python main.py --input input/photo.jpg \
    --person-model models/NanoDet/object_detection_nanodet_2022nov_int8.onnx
```

Or reduce input size in `src/config.py`:
```python
MAX_INPUT_WIDTH = 320  # Reduced from 640
MAX_INPUT_HEIGHT = 240  # Reduced from 480
```

#### 3. Slow Processing
```
FPS < 1
```

**Solutions:**
- Use INT8 quantized models
- Enable parallel pipeline
- Reduce input resolution
- Use lighter models (NanoDet, YuNet)

#### 4. No Detections
```
Detected 0 person(s) and 0 face(s)
```

**Solutions:**
- Lower confidence thresholds:
  ```bash
  python main.py --input input/photo.jpg \
      --person-threshold 0.3 \
      --face-threshold 0.3
  ```
- Try different models
- Check input image quality

#### 5. ONNX Runtime Error
```
Error: Failed to load ONNX model
```

**Solution:**
```bash
# Reinstall ONNX Runtime
pip install --upgrade onnxruntime

# Or use fallback detectors (automatic)
# The system will use HOG/Haar Cascade
```

---

## Examples

### Example 1: Process Multiple Images
```bash
#!/bin/bash
# process_folder.sh

INPUT_DIR="input"
OUTPUT_DIR="output"

for img in $INPUT_DIR/*.jpg $INPUT_DIR/*.png; do
    if [ -f "$img" ]; then
        echo "Processing: $img"
        python main.py --input "$img" --output-dir "$OUTPUT_DIR"
    fi
done

echo "All images processed!"
```

### Example 2: Compare Models
```bash
#!/bin/bash
# compare_models.sh

INPUT="input/test.jpg"

# Test NanoDet
python main.py --input $INPUT \
    --person-model models/NanoDet/object_detection_nanodet_2022nov.onnx \
    --output-dir output/nanodet/

# Test NanoDet-Plus
python main.py --input $INPUT \
    --person-model models/NanoDet-Plus/nanodet-plus-m_416.onnx \
    --output-dir output/nanodet_plus/

# Test RF-DETR
python main.py --input $INPUT \
    --person-model models/RF-DETR-Nano/model_int8.onnx \
    --output-dir output/rfdetr/

echo "Model comparison complete! Check output/ folders"
```

### Example 3: Video Surveillance Processing
```bash
#!/bin/bash
# surveillance.sh

VIDEO_DIR="surveillance_footage"
OUTPUT_BASE="output/surveillance"

for video in $VIDEO_DIR/*.mp4; do
    video_name=$(basename "$video" .mp4)
    echo "Processing surveillance video: $video_name"
    
    python main.py \
        --input "$video" \
        --pipeline sequential \
        --person-threshold 0.4 \
        --face-threshold 0.5 \
        --output-dir "$OUTPUT_BASE/$video_name/"
    
    echo "Completed: $video_name"
done
```

### Example 4: Programmatic Usage
```python
#!/usr/bin/env python3
"""Custom processing script"""

import cv2
from src.pipeline import create_pipeline
from src import config

# Create pipeline
pipeline = create_pipeline(
    pipeline_mode="sequential",
    person_model_path="models/NanoDet/object_detection_nanodet_2022nov.onnx",
    face_model_path="models/YuNet/face_detection_yunet_2023mar.onnx",
    person_threshold=0.5,
    face_threshold=0.5
)

# Load image
image = cv2.imread("input/photo.jpg")

# Process
pipeline.metrics_tracker.start_processing()
person_count, face_count = pipeline.process_image(image, "output/result")
pipeline.metrics_tracker.end_processing()

# Print results
print(f"Found {person_count} persons and {face_count} faces")
pipeline.metrics_tracker.print_summary()
```

---

## Configuration

### Editing Configuration

Edit `src/config.py` to change default settings:

```python
# Pipeline mode
DEFAULT_PIPELINE = "sequential"  # or "parallel"

# Model paths
PERSON_MODEL_PATH = "models/NanoDet/object_detection_nanodet_2022nov.onnx"
FACE_MODEL_PATH = "models/YuNet/face_detection_yunet_2023mar.onnx"

# Thresholds
PERSON_CONFIDENCE_THRESHOLD = 0.5
FACE_CONFIDENCE_THRESHOLD = 0.5

# Performance
MAX_INPUT_WIDTH = 640
MAX_INPUT_HEIGHT = 480
MAX_THREADS = 2

# Output
SAVE_CROPPED_IMAGES = True
SAVE_ANNOTATED_OUTPUT = True
SAVE_METRICS_TXT = True
```

---

## Tips & Best Practices

### 🎯 For Best Accuracy
- Use sequential pipeline
- Use higher resolution models (640 input)
- Set moderate thresholds (0.5-0.6)
- Use FP32 models (not quantized)

### ⚡ For Best Speed
- Use parallel pipeline
- Use INT8 quantized models
- Lower input resolution
- Use lightweight models (NanoDet, YuNet)

### 💾 For Low Memory
- Use quantized models (INT8)
- Reduce MAX_INPUT_WIDTH/HEIGHT
- Use sequential pipeline
- Set MAX_THREADS = 1

### 🎥 For Video Processing
- Use sequential pipeline (better tracking)
- Enable metric logging
- Process at lower FPS if needed
- Monitor memory usage

---

## Getting Help

### Documentation
- Pipeline Architecture: `docs/PIPELINE_ARCHITECTURE.md`
- API Reference: `docs/API_REFERENCE.md`
- Model Guide: `models/DOWNLOADED_MODELS.md`

### Support
- GitHub Issues: [Report bugs](https://github.com/your-repo/issues)
- Discussions: [Ask questions](https://github.com/your-repo/discussions)

### Community Examples
Check `example.py` for more usage patterns!

---

## Next Steps

1. ✅ [Download models](scripts/download_models.py)
2. ✅ [Run examples](example.py)
3. ✅ [Test your images](#basic-usage)
4. ✅ [Optimize performance](#advanced-usage)
5. ✅ [Read architecture docs](docs/PIPELINE_ARCHITECTURE.md)

Happy detecting! 🎉
