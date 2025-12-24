# Debug Report - Detected Human Faces System

**Date:** 24 Tháng 12, 2025  
**Status:** ✅ All Issues Resolved

---

## 🐛 Issues Found and Fixed

### Issue 1: Import Error
**Error:**
```
ImportError: cannot import name 'DetectionPipeline' from 'src.pipeline'
```

**Root Cause:**  
`src/__init__.py` was importing a non-existent class name `DetectionPipeline`. The actual class names in `src/pipeline.py` are:
- `ProcessingPipeline` (base class)
- `SequentialPipeline`
- `ParallelPipeline`

**Fix:**  
Updated `src/__init__.py` to import correct class names:
```python
from .pipeline import create_pipeline, ProcessingPipeline, SequentialPipeline, ParallelPipeline
```

**File Changed:** `src/__init__.py`

---

### Issue 2: FileNotFoundError for Logs
**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: '/home/baobao/sshfs/orange-pi/Projects/detected-human-faces/src/logs/test_summary.txt'
```

**Root Cause:**  
`BASE_DIR` in `src/config.py` was set to the `src/` directory instead of project root:
```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Points to src/
```

This caused `LOG_DIR = os.path.join(BASE_DIR, "logs")` to create path `/src/logs/` instead of `/logs/`.

**Fix:**  
Updated `src/config.py` to point to project root:
```python
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Project root
```

**File Changed:** `src/config.py`

---

### Issue 3: ONNX Dimension Mismatch
**Error:**
```
[ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Got invalid dimensions for input: input.1
 index: 2 Got: 640 Expected: 416
 index: 3 Got: 640 Expected: 416
```

**Root Cause:**  
Both `PersonDetector` and `FaceDetector` had hardcoded input sizes:
- PersonDetector: `input_size = (640, 640)`
- FaceDetector: `input_size = (320, 320)`

However, different models expect different input dimensions:
- NanoDet: 416x416
- NanoDet-Plus-320: 320x320
- NanoDet-Plus-416: 416x416
- YuNet: 320x320
- RF-DETR: Various sizes

**Fix:**  
Updated both `_detect_onnx()` methods to dynamically read input shape from model metadata:

```python
def _detect_onnx(self, image: np.ndarray) -> List[Detection]:
    # Get input shape from model
    input_shape = self.model.get_inputs()[0].shape
    # Handle dynamic dimensions
    if len(input_shape) >= 4:
        input_height = input_shape[2] if isinstance(input_shape[2], int) else 416
        input_width = input_shape[3] if isinstance(input_shape[3], int) else 416
    else:
        input_height, input_width = 416, 416
    
    input_size = (input_width, input_height)
    img_resized = cv2.resize(image, input_size)
    # ... rest of preprocessing
```

**Files Changed:** 
- `src/detector.py` (PersonDetector._detect_onnx)
- `src/detector.py` (FaceDetector._detect_onnx)

---

## ✅ Test Results After Fixes

### Complete Test Suite Execution
```bash
bash scripts/run_complete_test.sh input/test.png
```

**Results:**
- Total Tests: 18
- Passed: 18 ✅
- Failed: 0
- Success Rate: 100%

### Models Tested Successfully

#### Person Detection (10 tests)
- ✅ NanoDet-FP32 (sequential & parallel)
- ✅ NanoDet-INT8 (sequential & parallel)
- ✅ NanoDet-Plus-320 (sequential)
- ✅ NanoDet-Plus-416 (sequential)
- ✅ RF-DETR-Nano-FP32 (sequential)
- ✅ RF-DETR-Nano-FP16 (sequential)
- ✅ RF-DETR-Nano-INT8 (sequential)
- ✅ RF-DETR-Nano-Quantized (sequential)

#### Face Detection (8 tests)
- ✅ YuNet-FP32 (sequential & parallel)
- ✅ YuNet-INT8 (sequential & parallel)
- ✅ YOLOv8-Face (sequential & parallel)
- ✅ UltraFace-320 (sequential)
- ✅ UltraFace-640 (sequential)

---

## 📊 Performance Highlights

### Top 3 Fastest Models
1. **YuNet-INT8** - 0.50 FPS (sequential/parallel) - **Recommended for real-time**
2. **NanoDet-INT8** - 0.50 FPS (sequential)
3. **YuNet-FP32** - 0.33-0.50 FPS

### Smallest Models
1. **YuNet-INT8** - 0.09 MB
2. **YuNet-FP32** - 0.22 MB
3. **NanoDet-INT8** - 0.98 MB

### Best Detection Rate
1. **NanoDet-INT8** (parallel) - 6 faces detected
2. **NanoDet-FP32** (parallel) - 6 faces detected
3. **YOLOv8-Face** (parallel) - 1 face detected

---

## 📁 Generated Reports

All reports are in `docs/reports/test_run_20251224_124525/`:

- **Summary Report:** `test_run_20251224_124525_summary.md`
- **Performance Analysis:** `test_run_20251224_124525_performance_analysis.md`
- **Sequence Diagrams:** `test_run_20251224_124525_sequence_diagram.md`
- **CSV Results:** `test_run_20251224_124525_results.csv`

### View Reports
```bash
# Summary
cat docs/reports/test_run_20251224_124525_summary.md

# Performance Analysis
cat docs/reports/test_run_20251224_124525_performance_analysis.md

# Sequence Diagram
cat docs/reports/test_run_20251224_124525_sequence_diagram.md

# CSV Data
cat docs/reports/test_run_20251224_124525_results.csv
```

---

## 🎯 Recommendations

### For Real-Time Processing (Speed Priority)
```bash
python main.py --input input/video.mp4 \
  --person-model models/NanoDet/object_detection_nanodet_2022nov_int8.onnx \
  --face-model models/YuNet/face_detection_yunet_2023mar_int8.onnx \
  --pipeline parallel
```

### For Accuracy (Detection Quality Priority)
```bash
python main.py --input input/image.jpg \
  --person-model models/NanoDet/object_detection_nanodet_2022nov.onnx \
  --face-model models/YOLOv8-Face/yolov8n-face.onnx \
  --pipeline parallel
```

### For Low Memory Systems
```bash
python main.py --input input/image.jpg \
  --person-model models/NanoDet/object_detection_nanodet_2022nov_int8.onnx \
  --face-model models/YuNet/face_detection_yunet_2023mar_int8.onnx \
  --pipeline sequential
```

---

## 🔍 Debugging Tips

### If model fails to load:
```bash
# Check model file exists
ls -lh models/NanoDet/object_detection_nanodet_2022nov.onnx

# Check ONNX Runtime can load it
python -c "import onnxruntime; sess = onnxruntime.InferenceSession('models/NanoDet/object_detection_nanodet_2022nov.onnx'); print('Model loaded successfully')"
```

### If input dimension errors occur:
```bash
# Check model input shape
python -c "
import onnxruntime as ort
sess = ort.InferenceSession('models/NanoDet/object_detection_nanodet_2022nov.onnx')
for inp in sess.get_inputs():
    print(f'{inp.name}: {inp.shape}')
"
```

### If logs not saving:
```bash
# Verify LOG_DIR path
python -c "from src import config; print(f'LOG_DIR: {config.LOG_DIR}')"

# Check logs directory exists
ls -ld logs/
```

---

## 📝 Summary

All critical bugs have been resolved:
1. ✅ Import errors fixed with correct class names
2. ✅ File path issues resolved with proper BASE_DIR
3. ✅ ONNX dimension errors eliminated with dynamic shape detection

The system now successfully runs all 18 model configurations with 100% pass rate.

**System Status:** 🟢 Fully Operational
