# Pipeline Architecture Documentation

## 📋 Table of Contents
1. [System Overview](#system-overview)
2. [Pipeline Modes](#pipeline-modes)
3. [Component Architecture](#component-architecture)
4. [Data Flow](#data-flow)
5. [Performance Optimization](#performance-optimization)

---

## System Overview

This detection system is designed for **Orange Pi RV 2** with 4GB RAM, optimized for real-time person and face detection using ONNX models.

### Key Features
- ✅ Dual-mode pipeline (Sequential & Parallel)
- ✅ ONNX model support with CPU optimization
- ✅ Fallback detectors (HOG, Haar Cascade)
- ✅ Memory-efficient processing
- ✅ Comprehensive metrics tracking
- ✅ Support for images and videos

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Main Entry Point                      │
│                         (main.py)                            │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    Pipeline Factory                          │
│                 (create_pipeline())                          │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
┌──────────────────┐          ┌──────────────────┐
│  Sequential      │          │   Parallel       │
│  Pipeline        │          │   Pipeline       │
└────────┬─────────┘          └────────┬─────────┘
         │                             │
         └──────────┬──────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌──────────────┐        ┌─────────────┐
│   Person     │        │    Face     │
│   Detector   │        │   Detector  │
└──────────────┘        └─────────────┘
        │                       │
        └───────────┬───────────┘
                    │
                    ▼
            ┌──────────────┐
            │   Metrics    │
            │   Tracker    │
            └──────────────┘
```

---

## Pipeline Modes

### 1. Sequential Pipeline

**Workflow:** Person Detection → Face Detection (on person crops)

```
Input Image
    │
    ▼
┌─────────────────┐
│ Person Detector │  ← Detect all persons in full image
└────────┬────────┘
         │
         ▼
   [Person Crops]
         │
         ├─→ Person 1 Crop → ┌────────────────┐
         │                    │ Face Detector  │
         ├─→ Person 2 Crop → │  (on crop)     │
         │                    └────────────────┘
         └─→ Person N Crop →       │
                                   ▼
                            [Face Detections]
```

**Advantages:**
- ✅ Better accuracy (faces detected in context)
- ✅ Faster face detection (smaller search area)
- ✅ Associates faces with specific persons
- ✅ Lower memory usage

**Use Cases:**
- Crowded scenes
- Multiple persons per frame
- When person-face association is needed

### 2. Parallel Pipeline

**Workflow:** Person Detection || Face Detection (independent)

```
Input Image
    │
    ├─────────────────┬─────────────────┐
    │                 │                 │
    ▼                 ▼                 ▼
┌─────────┐    ┌─────────┐    ┌─────────┐
│ Person  │    │  Face   │    │ Metrics │
│Detector │    │Detector │    │ Tracker │
└────┬────┘    └────┬────┘    └─────────┘
     │              │
     └──────┬───────┘
            │
            ▼
    [Combined Results]
```

**Advantages:**
- ✅ Faster processing (parallel execution)
- ✅ Independent detections
- ✅ Good for sparse scenes

**Use Cases:**
- Single person scenarios
- When speed is critical
- When faces might be outside person boxes

---

## Component Architecture

### 1. Detector Components

#### BaseDetector (Abstract)
```python
class BaseDetector:
    - model_path: str
    - confidence_threshold: float
    - model: ONNXRuntime Session
    
    + load_model()
    + detect(image) → List[Detection]
```

#### PersonDetector
- **Primary:** ONNX model (YOLO/NanoDet/etc.)
- **Fallback:** HOG + SVM (OpenCV)
- **Input:** Full image (640x640)
- **Output:** Person bounding boxes

```python
PersonDetector:
    + _detect_onnx(image) → List[Detection]
    + _detect_hog(image) → List[Detection]
```

#### FaceDetector
- **Primary:** ONNX model (YuNet/RetinaFace/etc.)
- **Fallback:** Haar Cascade (OpenCV)
- **Input:** Person crop or full image
- **Output:** Face bounding boxes

```python
FaceDetector:
    + _detect_onnx(image) → List[Detection]
    + _detect_cascade(image) → List[Detection]
```

### 2. Pipeline Components

#### ProcessingPipeline (Abstract)
```python
class ProcessingPipeline:
    - person_detector: PersonDetector
    - face_detector: FaceDetector
    - metrics_tracker: MetricsTracker
    
    + process_image(image, output_prefix) → (person_count, face_count)
    + process_video(video_path, output_prefix)
```

#### SequentialPipeline
```python
SequentialPipeline(ProcessingPipeline):
    + process_image():
        1. Detect persons in full image
        2. For each person crop:
            a. Detect faces
            b. Save crops
            c. Track metrics
```

#### ParallelPipeline
```python
ParallelPipeline(ProcessingPipeline):
    + process_image():
        1. ThreadPoolExecutor (2 threads)
        2. Thread 1: Person detection
        3. Thread 2: Face detection
        4. Merge results
```

### 3. Metrics Components

#### MetricsTracker
```python
MetricsTracker:
    - fps_values: List[float]
    - accuracy_values: List[float]
    - person_processing_times: List[float]
    
    + start_processing()
    + end_processing()
    + add_frame_metrics(fps, accuracy, time)
    + get_summary() → Dict
    + print_summary()
    + save_to_file(path)
```

---

## Data Flow

### Sequential Pipeline Flow

```
┌─────────────┐
│ Input Image │
│  (HxWx3)    │
└──────┬──────┘
       │
       ▼
┌──────────────────────┐
│  Resize if needed    │  MAX_WIDTH=640, MAX_HEIGHT=480
│  (Memory opt)        │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Person Detection    │
│  - Preprocess        │  ① RGB, Normalize, Transpose
│  - ONNX Inference    │  ② model.run(input)
│  - NMS               │  ③ Non-max suppression
│  - Threshold filter  │  ④ confidence >= 0.5
└──────┬───────────────┘
       │
       ▼
   [Person_1] [Person_2] ... [Person_N]
       │
       ├─→ ┌──────────────────────┐
       │   │ Crop Person ROI      │
       │   └──────┬───────────────┘
       │          │
       │          ▼
       │   ┌──────────────────────┐
       │   │  Face Detection      │
       │   │  (on person crop)    │
       │   └──────┬───────────────┘
       │          │
       │          ▼
       │      [Face_1] [Face_2]
       │          │
       │          ▼
       │   ┌──────────────────────┐
       │   │ Save Crops           │
       │   │ - person_0.jpg       │
       │   │ - face_0_0.jpg       │
       │   │ - face_0_1.jpg       │
       │   └──────────────────────┘
       │
       └─→ (Repeat for each person)
              │
              ▼
       ┌──────────────────────┐
       │  Metrics Tracking    │
       │  - FPS calculation   │
       │  - Processing time   │
       │  - Accuracy metrics  │
       └──────────────────────┘
```

### Parallel Pipeline Flow

```
┌─────────────┐
│ Input Image │
└──────┬──────┘
       │
       ├─────────────────┬─────────────────┐
       │                 │                 │
       ▼                 ▼                 │
┌──────────────┐  ┌──────────────┐        │
│   Thread 1   │  │   Thread 2   │        │
│              │  │              │        │
│   Person     │  │    Face      │        │
│  Detection   │  │  Detection   │        │
│              │  │              │        │
│ [Person_1]   │  │  [Face_1]    │        │
│ [Person_2]   │  │  [Face_2]    │        │
└──────┬───────┘  └──────┬───────┘        │
       │                 │                │
       └────────┬────────┘                │
                │                         │
                ▼                         ▼
         ┌──────────────┐        ┌──────────────┐
         │ Save Results │        │   Metrics    │
         └──────────────┘        └──────────────┘
```

---

## Performance Optimization

### 1. Memory Optimization

**Input Resizing:**
```python
MAX_INPUT_WIDTH = 640
MAX_INPUT_HEIGHT = 480

if w > MAX_WIDTH or h > MAX_HEIGHT:
    scale = min(MAX_WIDTH/w, MAX_HEIGHT/h)
    image = cv2.resize(image, (new_w, new_h))
```

**Model Input Size:**
- Person detector: 640x640
- Face detector: 320x320

### 2. CPU Optimization

**ONNX Runtime Settings:**
```python
providers=['CPUExecutionProvider']  # CPU-only for Orange Pi
```

**Threading:**
```python
MAX_THREADS = 2  # Limited for 4GB RAM
ThreadPoolExecutor(max_workers=MAX_THREADS)
```

### 3. Processing Optimization

**Batch Processing:** Disabled (memory constraint)

**NMS (Non-Maximum Suppression):**
```python
cv2.dnn.NMSBoxes(boxes, scores, score_threshold, nms_threshold)
```

**Early Stopping:**
```python
if confidence < threshold:
    continue  # Skip low-confidence detections
```

### 4. Expected Performance

| Configuration | FPS (Sequential) | FPS (Parallel) | Memory Usage |
|--------------|------------------|----------------|--------------|
| 640x480 Image | 2-5 FPS | 3-7 FPS | ~800MB |
| Video (640x480) | 1-3 FPS | 2-5 FPS | ~1.2GB |
| Multiple Persons | 0.5-2 FPS | 1-3 FPS | ~1.5GB |

---

## Configuration Parameters

### Detection Thresholds
```python
PERSON_CONFIDENCE_THRESHOLD = 0.5
FACE_CONFIDENCE_THRESHOLD = 0.5
```

### Pipeline Selection
```python
DEFAULT_PIPELINE = "sequential"  # or "parallel"
```

### Thread Configuration
```python
USE_THREADING = True
MAX_THREADS = 2
```

### Output Settings
```python
SAVE_CROPPED_IMAGES = True
SAVE_ANNOTATED_OUTPUT = True
SAVE_METRICS_TXT = True
```

---

## Error Handling & Fallbacks

### Model Loading Failure
```
ONNX Model Load Failed
        ↓
Use Fallback Detector
        ↓
Person: HOG + SVM
Face: Haar Cascade
```

### Processing Errors
```python
try:
    # Process with ONNX
except Exception:
    # Fallback to OpenCV detectors
```

### Memory Errors
```python
# Automatic input resizing
# Limited threading (MAX_THREADS=2)
# No batch processing
```

---

## Extension Points

### Adding New Models
1. Place ONNX model in `models/` folder
2. Update `config.py`:
   ```python
   PERSON_MODEL_PATH = "models/new_model.onnx"
   ```
3. Run with `--person-model` flag

### Custom Pipeline
1. Inherit from `ProcessingPipeline`
2. Implement `process_image()` method
3. Register in `create_pipeline()` factory

### Custom Metrics
1. Extend `MetricsTracker` class
2. Add custom metric collection
3. Update summary generation

---

## Debugging & Monitoring

### Log Files
```
logs/
├── {input_name}_summary.txt
├── person_detection_metrics.log
└── face_detection_metrics.log
```

### Output Files
```
output/
├── {input_name}_person_0.jpg
├── {input_name}_person_1.jpg
├── {input_name}_face_0_0.jpg
└── {input_name}_annotated.jpg
```

### Metrics Displayed
- Total processing time
- FPS (min/max/avg)
- Detection accuracy
- Per-person processing time

---

## Best Practices

### For Sequential Pipeline
✅ Use when: Multiple persons in scene
✅ Set: Lower person threshold (0.3-0.4)
✅ Optimize: Face detection on crops

### For Parallel Pipeline
✅ Use when: Single person scenarios
✅ Set: Higher thresholds (0.5-0.7)
✅ Optimize: Both detectors independently

### Memory Management
✅ Resize large inputs (>640x480)
✅ Limit concurrent threads (MAX_THREADS=2)
✅ Clear crops after processing

### Model Selection
✅ Lightweight models: YuNet, NanoDet, UltraFace
✅ Quantized models: INT8 for faster inference
✅ Test fallback: Ensure HOG/Cascade work

---

## Conclusion

This pipeline architecture provides:
- 🚀 **Flexibility:** Two processing modes
- ⚡ **Performance:** Optimized for Orange Pi
- 🛡️ **Reliability:** Fallback detectors
- 📊 **Monitoring:** Comprehensive metrics
- 🔧 **Extensibility:** Easy to customize

For usage instructions, see [USER_GUIDE.md](USER_GUIDE.md)
