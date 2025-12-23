# Human and Face Detection System

Hệ thống phát hiện người và khuôn mặt được tối ưu hóa cho **Orange Pi RV 2** (4GB RAM).

## 🌟 Tính năng

- ✅ Hỗ trợ đầu vào: **ảnh** hoặc **video**
- ✅ Tùy chọn model ONNX để sử dụng
- ✅ **2 pipeline xử lý**:
  - **Sequential**: Phát hiện người trước → phát hiện khuôn mặt sau
  - **Parallel**: Phát hiện người và khuôn mặt song song
- ✅ **Báo cáo hiệu suất chi tiết**:
  - Thời gian xử lý mỗi người
  - Thời gian xử lý tổng
  - FPS cao nhất, thấp nhất, trung bình
  - Accuracy min, max, avg
- ✅ Mỗi ảnh crop có file `.txt` đi kèm chứa FPS và Accuracy
- ✅ Tối ưu hóa cho thiết bị nhúng (RAM thấp)

## 📁 Cấu trúc thư mục

```
detected-human-faces/
├── config.py           # Cấu hình hệ thống
├── detector.py         # Các class phát hiện (PersonDetector, FaceDetector)
├── pipeline.py         # Pipeline xử lý (Sequential, Parallel)
├── metrics.py          # Theo dõi và tính toán metrics
├── main.py             # File chính để chạy chương trình
├── requirements.txt    # Danh sách thư viện cần thiết
├── models/             # Thư mục chứa các model ONNX
├── input/              # Thư mục chứa ảnh/video đầu vào
├── output/             # Thư mục chứa kết quả xử lý
└── logs/               # Thư mục chứa file báo cáo
```

## 🚀 Cài đặt

### 1. Yêu cầu hệ thống

- Orange Pi RV 2 (4GB RAM) hoặc thiết bị tương tự
- Python 3.8+
- OpenCV, NumPy, ONNX Runtime

### 2. Cài đặt thư viện

```bash
cd detected-human-faces
pip install -r requirements.txt
```

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
