# Quick Test Guide - Detected Human Faces

## 🚀 Chạy Test Đầy Đủ (All Models)

```bash
# Chạy tất cả các models và tạo reports
bash scripts/run_complete_test.sh input/test.png

# Hoặc với video
bash scripts/run_complete_test.sh input/video.mp4
```

**Kết quả sẽ được lưu tại:**
- 📊 Reports: `docs/reports/test_run_*_summary.md`
- 📈 Performance: `docs/reports/test_run_*_performance_analysis.md`
- 📉 Sequence Diagrams: `docs/reports/test_run_*_sequence_diagram.md`
- 📁 CSV: `docs/reports/test_run_*_results.csv`
- 📝 Logs: `logs/test_run_*_master.log`

---

## ⚡ Test Một Model Cụ Thể

### Test Model Nhanh Nhất (YuNet-INT8)
```bash
python main.py \
  --input input/test.png \
  --output-dir output/yunet_test \
  --face-model models/YuNet/face_detection_yunet_2023mar_int8.onnx \
  --pipeline sequential
```

### Test Model Chính Xác (YOLOv8-Face)
```bash
python main.py \
  --input input/test.png \
  --output-dir output/yolov8_test \
  --face-model models/YOLOv8-Face/yolov8n-face.onnx \
  --pipeline parallel
```

### Test Person Detection (NanoDet)
```bash
python main.py \
  --input input/test.png \
  --output-dir output/nanodet_test \
  --person-model models/NanoDet/object_detection_nanodet_2022nov_int8.onnx \
  --pipeline sequential
```

---

## 🔍 Xem Kết Quả Test

### Xem Summary Report
```bash
# Tìm test run mới nhất
ls -lt docs/reports/ | grep summary | head -1

# Xem nội dung
cat docs/reports/test_run_YYYYMMDD_HHMMSS_summary.md
```

### Xem Performance Analysis
```bash
cat docs/reports/test_run_YYYYMMDD_HHMMSS_performance_analysis.md
```

### Xem CSV Results
```bash
# Xem dạng table
column -t -s',' docs/reports/test_run_YYYYMMDD_HHMMSS_results.csv | less -S

# Hoặc import vào Excel/LibreOffice
```

### Xem Sequence Diagrams
```bash
cat docs/reports/test_run_YYYYMMDD_HHMMSS_sequence_diagram.md
```

---

## 📊 So Sánh Models

### Top Models theo Speed
```bash
grep "FPS" docs/reports/test_run_*_summary.md | sort -k3 -rn | head -5
```

### Top Models theo Size (Smallest)
```bash
grep "Size" docs/reports/test_run_*_summary.md | sort -k3 -n | head -5
```

### Models với Detection Cao Nhất
```bash
grep "Detections" docs/reports/test_run_*_summary.md | sort -k3 -rn | head -5
```

---

## 🛠️ Debug và Troubleshooting

### Kiểm tra Model Load được không
```bash
python -c "
import onnxruntime as ort
model_path = 'models/YuNet/face_detection_yunet_2023mar_int8.onnx'
sess = ort.InferenceSession(model_path)
print(f'✓ Model loaded: {model_path}')
print(f'Input shape: {sess.get_inputs()[0].shape}')
"
```

### Xem Log Chi Tiết của Model
```bash
# Tìm log file
ls -lt logs/ | grep test_run | head -5

# Xem nội dung
cat logs/test_run_YYYYMMDD_HHMMSS_ModelName_pipeline.log
```

### Kiểm tra Lỗi trong Test
```bash
# Tìm tất cả logs có error
grep -r "Error\|Exception\|FAILED" logs/test_run_*

# Xem log master
cat logs/test_run_YYYYMMDD_HHMMSS_master.log
```

---

## 📈 Performance Benchmarks (Reference)

### ⚡ Fastest Models (Orange Pi RV 2)
| Model | FPS | Size | Pipeline | Use Case |
|-------|-----|------|----------|----------|
| YuNet-INT8 | 0.50 | 0.09 MB | Sequential | Real-time face |
| NanoDet-INT8 | 0.50 | 0.98 MB | Sequential | Real-time person |
| YuNet-FP32 | 0.33 | 0.22 MB | Sequential | Fast face |

### 🎯 Most Accurate Models
| Model | Detections | Size | Pipeline | Use Case |
|-------|-----------|------|----------|----------|
| NanoDet-FP32 | 6 faces | 3.62 MB | Parallel | High accuracy |
| NanoDet-INT8 | 6 faces | 0.98 MB | Parallel | Balance speed/accuracy |
| YOLOv8-Face | 1 face | 11.68 MB | Parallel | Robust detection |

### 💾 Smallest Models
| Model | Size | FPS | Pipeline | Use Case |
|-------|------|-----|----------|----------|
| YuNet-INT8 | 0.09 MB | 0.50 | Sequential | Embedded systems |
| YuNet-FP32 | 0.22 MB | 0.33 | Sequential | Low memory |
| NanoDet-INT8 | 0.98 MB | 0.50 | Sequential | IoT devices |

---

## 🎯 Recommended Configurations

### 1. Real-Time Processing (Speed Priority)
```bash
python main.py \
  --input input/webcam.mp4 \
  --person-model models/NanoDet/object_detection_nanodet_2022nov_int8.onnx \
  --face-model models/YuNet/face_detection_yunet_2023mar_int8.onnx \
  --pipeline parallel \
  --output-dir output/realtime
```

### 2. High Accuracy (Quality Priority)
```bash
python main.py \
  --input input/photo.jpg \
  --person-model models/NanoDet/object_detection_nanodet_2022nov.onnx \
  --face-model models/YOLOv8-Face/yolov8n-face.onnx \
  --pipeline parallel \
  --output-dir output/highquality
```

### 3. Low Memory (Embedded Priority)
```bash
python main.py \
  --input input/video.mp4 \
  --person-model models/NanoDet/object_detection_nanodet_2022nov_int8.onnx \
  --face-model models/YuNet/face_detection_yunet_2023mar_int8.onnx \
  --pipeline sequential \
  --output-dir output/lowmem
```

---

## 📝 Logs và Reports Structure

```
logs/
├── test_run_20251224_124525_master.log          # Master log
├── test_run_20251224_124525_NanoDet-FP32_sequential.log
├── test_run_20251224_124525_NanoDet-FP32_parallel.log
└── ...

docs/reports/
├── test_run_20251224_124525_summary.md          # Summary
├── test_run_20251224_124525_performance_analysis.md  # Analysis
├── test_run_20251224_124525_sequence_diagram.md # Diagrams
└── test_run_20251224_124525_results.csv         # CSV data

output/
└── test_run_20251224_124525/
    ├── NanoDet-FP32_sequential/
    │   ├── test_person_0.jpg
    │   └── test_face_0_0.jpg
    └── ...
```

---

## 🚦 Quick Status Check

```bash
# Xem test runs gần đây
ls -lt docs/reports/ | grep summary | head -5

# Xem success rate
tail -20 logs/test_run_*_master.log | grep "Passed\|Failed"

# Số lượng models đã test
ls -1 models/*/*.onnx 2>/dev/null | wc -l
```

---

## 📚 Chi Tiết Hơn

- **Full Documentation:** `docs/USER_GUIDE.md`
- **Architecture:** `docs/PIPELINE_ARCHITECTURE.md`
- **Debug Report:** `DEBUG_REPORT.md`
- **Quick Commands:** `bash QUICK_START.sh`

---

## ✅ Verification Checklist

- [x] All imports working correctly
- [x] Paths configured properly (BASE_DIR fixed)
- [x] Dynamic ONNX input shape detection
- [x] 18/18 tests passing
- [x] Reports generating successfully
- [x] Sequence diagrams created
- [x] Performance analysis complete

**System Status:** 🟢 Fully Operational
