# Human and Face Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-orange.svg)](https://onnxruntime.ai)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Optimized for Orange Pi RV 2 (4GB RAM)** - A complete human and face detection system with dual-pipeline processing, comprehensive metrics, and extensive model support.

---

## 🌟 Key Features

- ✅ **Dual Pipeline Modes**
  - **Sequential**: Person detection → Face detection (on person crops)
  - **Parallel**: Person & Face detection simultaneously
- ✅ **Extensive Model Support**
  - 13+ ONNX models included
  - Person detection: NanoDet, RF-DETR, EfficientDet-Lite
  - Face detection: YuNet, YOLOv8-Face, UltraFace
- ✅ **Comprehensive Testing Suite**
  - Automated model testing
  - Performance benchmarking
  - Detailed reports with sequence diagrams
- ✅ **Performance Optimized**
  - Memory-efficient processing (<1GB)
  - CPU-only inference (ONNX Runtime)
  - Quantized models (INT8) support
- ✅ **Rich Documentation**
  - User guides with examples
  - Architecture documentation
  - Auto-generated test reports

---

## 📁 Project Structure

```
detected-human-faces/
├── src/                      # Source code package
│   ├── config.py            # Configuration settings
│   ├── detector.py          # Detection classes
│   ├── pipeline.py          # Pipeline implementations
│   └── metrics.py           # Performance tracking
├── scripts/                  # Utility scripts
│   ├── download_models.py   # Model downloader
│   ├── test_all_models.sh   # Complete test suite
│   ├── analyze_logs.py      # Log analyzer
│   └── run_complete_test.sh # Master test runner
├── docs/                     # Documentation
│   ├── README.md            # Documentation index
│   ├── USER_GUIDE.md        # User guide
│   ├── PIPELINE_ARCHITECTURE.md  # Technical docs
│   └── reports/             # Auto-generated reports
├── models/                   # ONNX models (13+ models)
│   ├── NanoDet/
│   ├── YuNet/
│   ├── RF-DETR-Nano/
│   └── ...
├── input/                    # Input images/videos
├── output/                   # Detection results
├── logs/                     # Performance logs
├── main.py                   # Main entry point
├── example.py                # Example usage
└── requirements.txt          # Python dependencies
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd detected-human-faces

# Install dependencies
pip install -r requirements.txt

# Download models (281MB+)
python scripts/download_models.py
```

### 2. Basic Usage

```bash
# Process an image
python main.py --input input/photo.jpg

# Process a video
python main.py --input input/video.mp4

# Use parallel pipeline for speed
python main.py --input input/photo.jpg --pipeline parallel
```

### 3. Test All Models

```bash
# Run comprehensive test suite
bash scripts/run_complete_test.sh input/test.png
```

**This will automatically:**
- ✅ Test all 19+ model configurations
- ✅ Generate performance reports
- ✅ Create sequence diagrams
- ✅ Analyze and rank models
- ✅ Save results to `docs/reports/`

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [User Guide](docs/USER_GUIDE.md) | Complete usage guide with examples |
| [Pipeline Architecture](docs/PIPELINE_ARCHITECTURE.md) | Technical architecture & design |
| [Documentation Index](docs/README.md) | All documentation overview |
| [Model Documentation](models/DOWNLOADED_MODELS.md) | Available models & usage |

---

## 🎯 Usage Examples

### Example 1: Default Configuration
```bash
python main.py --input input/photo.jpg
```

### Example 2: Custom Models
```bash
python main.py \
    --input input/photo.jpg \
    --person-model models/NanoDet/object_detection_nanodet_2022nov.onnx \
    --face-model models/YuNet/face_detection_yunet_2023mar.onnx
```

### Example 3: Optimized for Speed
```bash
python main.py \
    --input input/video.mp4 \
    --person-model models/NanoDet/object_detection_nanodet_2022nov_int8.onnx \
    --face-model models/YuNet/face_detection_yunet_2023mar_int8.onnx \
    --pipeline parallel
```

### Example 4: High Accuracy Mode
```bash
python main.py \
    --input input/photo.jpg \
    --person-model models/RF-DETR-Nano/model.onnx \
    --face-model models/YOLOv8-Face/yolov8n-face.onnx \
    --pipeline sequential \
    --person-threshold 0.6 \
    --face-threshold 0.6
```

### Example 5: Programmatic Usage
```python
from src.pipeline import create_pipeline
import cv2

# Create pipeline
pipeline = create_pipeline(
    pipeline_mode="sequential",
    person_model_path="models/NanoDet/object_detection_nanodet_2022nov.onnx",
    face_model_path="models/YuNet/face_detection_yunet_2023mar.onnx"
)

# Process image
image = cv2.imread("input/photo.jpg")
person_count, face_count = pipeline.process_image(image, "output/result")

print(f"Detected {person_count} persons and {face_count} faces")
```

---

## 🔧 Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input`, `-i` | *required* | Input image or video file |
| `--pipeline`, `-p` | `sequential` | Pipeline mode: `sequential` or `parallel` |
| `--person-model` | `models/person_detection.onnx` | Person detection model path |
| `--face-model` | `models/face_detection.onnx` | Face detection model path |
| `--person-threshold` | `0.5` | Person detection confidence (0-1) |
| `--face-threshold` | `0.5` | Face detection confidence (0-1) |
| `--output-dir` | `output/` | Output directory for results |

---

## 📊 Available Models

### Person Detection Models (7 models)
| Model | Size | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| NanoDet-INT8 | 1MB | ⚡⚡⚡⚡ | ⭐⭐ | Real-time |
| NanoDet-FP32 | 3.6MB | ⚡⚡⚡ | ⭐⭐⭐ | Balanced |
| NanoDet-Plus | 4.6MB | ⚡⚡⚡ | ⭐⭐⭐ | Enhanced |
| RF-DETR-INT8 | 27MB | ⚡⚡ | ⭐⭐⭐ | Accurate |
| RF-DETR-FP32 | 103MB | ⚡ | ⭐⭐⭐⭐ | High accuracy |

### Face Detection Models (6 models)
| Model | Size | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| YuNet-INT8 | 99KB | ⚡⚡⚡⚡ | ⭐⭐ | Ultra-fast |
| YuNet-FP32 | 228KB | ⚡⚡⚡ | ⭐⭐⭐ | Fast |
| UltraFace-320 | 1.3MB | ⚡⚡⚡ | ⭐⭐ | Lightweight |
| YOLOv8-Face | 12MB | ⚡⚡ | ⭐⭐⭐ | Balanced |

See [Model Documentation](models/DOWNLOADED_MODELS.md) for complete list.

---

## 🧪 Testing & Benchmarking

### Run Complete Test Suite
```bash
bash scripts/run_complete_test.sh input/test.png
```

### What Gets Tested
- ✅ All person detection models
- ✅ All face detection models  
- ✅ Both pipeline modes (sequential & parallel)
- ✅ Performance metrics (FPS, accuracy, time)
- ✅ Detection counts

### Generated Reports
- **Summary Report**: Test results table, statistics, top performers
- **Performance Analysis**: Detailed metrics, comparisons, recommendations
- **Sequence Diagrams**: Visual flow diagrams (Mermaid format)
- **CSV Results**: Raw data for further analysis

**Example Output:**
```
docs/reports/
├── test_run_20251224_143052_summary.md
├── test_run_20251224_143052_performance_analysis.md
├── test_run_20251224_143052_sequence_diagram.md
└── test_run_20251224_143052_results.csv
```

---

## 📈 Performance Metrics

### Typical Performance (Orange Pi RV 2)

| Configuration | FPS | Memory | Use Case |
|--------------|-----|--------|----------|
| NanoDet-INT8 + YuNet-INT8 (Parallel) | 5-7 | ~800MB | Real-time |
| NanoDet + YuNet (Sequential) | 3-5 | ~1GB | Balanced |
| RF-DETR + YOLOv8 (Sequential) | 1-2 | ~1.5GB | High accuracy |

### Metrics Tracked
- ⏱️ Processing time per frame/image
- 📊 FPS (min, max, average)
- 🎯 Detection accuracy
- 👥 Person count
- 👤 Face count
- 💾 Memory usage

---

## 🎨 Output Files

### Directory Structure
```
output/
├── photo_person_0.jpg          # Cropped person 0
├── photo_person_1.jpg          # Cropped person 1
├── photo_face_0_0.jpg          # Face 0 from person 0
├── photo_face_0_1.jpg          # Face 1 from person 0
└── photo_annotated.jpg         # Original with boxes

logs/
└── photo_summary.txt           # Performance metrics
```

### Example Summary
```
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

---

## 🛠️ Configuration

Edit `src/config.py` to customize:

```python
# Pipeline mode
DEFAULT_PIPELINE = "sequential"  # or "parallel"

# Model paths
PERSON_MODEL_PATH = "models/NanoDet/object_detection_nanodet_2022nov.onnx"
FACE_MODEL_PATH = "models/YuNet/face_detection_yunet_2023mar.onnx"

# Detection thresholds
PERSON_CONFIDENCE_THRESHOLD = 0.5
FACE_CONFIDENCE_THRESHOLD = 0.5

# Performance optimization
MAX_INPUT_WIDTH = 640
MAX_INPUT_HEIGHT = 480
MAX_THREADS = 2

# Output settings
SAVE_CROPPED_IMAGES = True
SAVE_ANNOTATED_OUTPUT = True
SAVE_METRICS_TXT = True
```

---

## 🔍 Troubleshooting

### Common Issues

**Out of Memory:**
```bash
# Use quantized models
python main.py --input input/photo.jpg \
    --person-model models/NanoDet/object_detection_nanodet_2022nov_int8.onnx
```

**Slow Processing:**
```bash
# Enable parallel pipeline
python main.py --input input/photo.jpg --pipeline parallel
```

**No Detections:**
```bash
# Lower thresholds
python main.py --input input/photo.jpg \
    --person-threshold 0.3 --face-threshold 0.3
```

See [User Guide - Troubleshooting](docs/USER_GUIDE.md#troubleshooting) for more solutions.

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- **OpenCV Zoo** - Pre-trained models
- **ONNX Runtime** - Efficient inference
- **Hugging Face** - Model hosting
- **Orange Pi Community** - Hardware support

---

## 📧 Support

- 📖 **Documentation**: [docs/](docs/)
- 🐛 **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)

---

## 🎯 Roadmap

- [ ] Add GPU support (CUDA, OpenCL)
- [ ] Web interface for easy testing
- [ ] Docker containerization
- [ ] More model formats (TensorRT, TFLite)
- [ ] Real-time video streaming
- [ ] Face recognition (after detection)

---

**Made with ❤️ for Orange Pi RV 2**

*Last Updated: December 24, 2025*

### 3. Chuẩn bị models

Đặt các model ONNX vào thư mục `models/`:
- `person_detection.onnx` - Model phát hiện người
- `face_detection.onnx` - Model phát hiện khuôn mặt

**Lưu ý**: Nếu không có model ONNX, hệ thống sẽ tự động sử dụng:
- **HOG + SVM** cho phát hiện người
- **Haar Cascade** cho phát hiện khuôn mặt

#### Cách export model ONNX từ YOLOv8:

```python
from ultralytics import YOLO

# Export person detection model
model = YOLO('yolov8n.pt')
model.export(format='onnx', simplify=True)

# Export face detection model
face_model = YOLO('yolov8n-face.pt')
face_model.export(format='onnx', simplify=True)
```

## 📖 Sử dụng

### Xử lý ảnh với Sequential Pipeline

```bash
python main.py --input input/photo.jpg --pipeline sequential
```

### Xử lý video với Parallel Pipeline

```bash
python main.py --input input/video.mp4 --pipeline parallel
```

### Sử dụng custom models

```bash
python main.py --input input/test.jpg \
  --person-model models/yolov8n.onnx \
  --face-model models/yolov8n-face.onnx
```

### Điều chỉnh ngưỡng confidence

```bash
python main.py --input input/test.jpg \
  --person-threshold 0.6 \
  --face-threshold 0.7
```

### Xem tất cả tùy chọn

```bash
python main.py --help
```

## 📊 Kết quả đầu ra

### 1. Ảnh crop

**Sequential Pipeline**:
- `output/{filename}_person_0.jpg` - Ảnh người thứ 0
- `output/{filename}_person_0_face_0.jpg` - Khuôn mặt thứ 0 của người thứ 0
- `output/{filename}_person_0_face_1.jpg` - Khuôn mặt thứ 1 của người thứ 0

**Parallel Pipeline**:
- `output/{filename}_person_0.jpg` - Ảnh người thứ 0
- `output/{filename}_face_0.jpg` - Khuôn mặt thứ 0 (từ ảnh gốc)

### 2. File metrics (.txt)

Mỗi ảnh crop có file `.txt` đi kèm:

```
fps: 25.43
acc: 0.8756
```

### 3. Báo cáo tổng hợp

File `logs/{filename}_summary.txt` chứa:

```
PERFORMANCE REPORT
============================================================
Total frames processed: 150
Total processing time: 12.45s
Average person processing time: 82.45ms

FPS Statistics:
  Max FPS: 28.50
  Min FPS: 18.23
  Avg FPS: 24.15

Accuracy Statistics:
  Max Accuracy: 0.9512
  Min Accuracy: 0.7234
  Avg Accuracy: 0.8678
============================================================
```

## ⚙️ Cấu hình

Chỉnh sửa file `config.py` để thay đổi:

- Đường dẫn models
- Ngưỡng confidence
- Kích thước input tối đa
- Số threads xử lý
- Pipeline mode mặc định

```python
# Ví dụ trong config.py
DEFAULT_PIPELINE = PIPELINE_SEQUENTIAL
PERSON_CONFIDENCE_THRESHOLD = 0.5
FACE_CONFIDENCE_THRESHOLD = 0.5
MAX_INPUT_WIDTH = 640
MAX_INPUT_HEIGHT = 480
MAX_THREADS = 2  # Giới hạn cho 4GB RAM
```

## 🔧 Pipeline Modes

### Sequential Pipeline (Tuần tự)
1. Phát hiện tất cả người trong ảnh
2. Với mỗi người được phát hiện:
   - Crop ảnh người
   - Phát hiện khuôn mặt trong crop
   - Lưu các khuôn mặt tìm được

**Ưu điểm**: Chính xác hơn (face detection chỉ trong vùng person)  
**Nhược điểm**: Chậm hơn

### Parallel Pipeline (Song song)
1. Phát hiện người và khuôn mặt cùng lúc trên ảnh gốc
2. Lưu tất cả detections

**Ưu điểm**: Nhanh hơn  
**Nhược điểm**: Có thể phát hiện face ngoài vùng person

## 🎯 Tối ưu hóa cho Orange Pi

- Giới hạn kích thước input (640x480)
- Sử dụng CPU-only inference
- Giới hạn số threads (2 threads)
- Sử dụng ONNX Runtime cho hiệu suất tốt
- Fallback detectors (HOG, Haar) nếu không có ONNX

## 📝 Ví dụ workflow

```bash
# 1. Đặt ảnh vào thư mục input
cp my_photo.jpg input/

# 2. Chạy detection với sequential pipeline
python main.py --input input/my_photo.jpg --pipeline sequential

# 3. Kiểm tra kết quả
ls output/my_photo_*
# output/my_photo_person_0.jpg
# output/my_photo_person_0.txt
# output/my_photo_person_0_face_0.jpg
# output/my_photo_person_0_face_0.txt

# 4. Xem báo cáo
cat logs/my_photo_summary.txt
```

## 🐛 Troubleshooting

### Lỗi: Model not found
```
Warning: Model not found at models/person_detection.onnx
Using fallback cascade/HOG detector
```
**Giải pháp**: Hệ thống tự động sử dụng fallback detectors (HOG/Haar Cascade)

### Lỗi: Out of memory
**Giải pháp**: Giảm `MAX_INPUT_WIDTH` và `MAX_INPUT_HEIGHT` trong `config.py`

### Lỗi: Slow processing
**Giải pháp**: 
- Sử dụng parallel pipeline
- Giảm resolution input
- Tăng confidence threshold để giảm số detections

## 📄 License

MIT License

## 👨‍💻 Author

Created for Orange Pi RV 2 optimization
