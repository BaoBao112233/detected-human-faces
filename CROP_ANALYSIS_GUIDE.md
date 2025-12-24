# Phân Tích Chi Tiết Video - Hướng Dẫn Sử Dụng

## 🎯 Chức năng

Script `run_detailed_analysis.py` sẽ phân tích video và tạo ra:

### Cho mỗi Person được phát hiện:
- ✅ **Ảnh crop** của person (`frame_XXXX_person_Y.jpg`)
- ✅ **File thông số** chi tiết (`frame_XXXX_person_Y.txt`) chứa:
  - Số đối tượng (1 person)
  - Thời gian xử lý (ms)
  - Inference time (ms)
  - Accuracy (confidence score)
  - Kích thước ảnh crop
  - Kích thước frame gốc
  - FPS tại thời điểm phát hiện

### Cho mỗi Face được phát hiện:
- ✅ **Ảnh crop** của face (`frame_XXXX_person_Y_face_Z.jpg`)
- ✅ **File thông số** chi tiết (`frame_XXXX_person_Y_face_Z.txt`) chứa:
  - Số đối tượng (1 face)
  - Thời gian xử lý (ms)
  - Inference time (ms)
  - Accuracy (confidence score)
  - Kích thước ảnh crop face
  - Kích thước person crop
  - Kích thước frame gốc
  - FPS tại thời điểm phát hiện

### File tổng hợp:
- 📄 **detailed_analysis.txt** - Log đầy đủ của tất cả frames

## 🚀 Cách Sử dụng

### Option 1: Chạy script Python trực tiếp

```bash
python run_detailed_analysis.py --input input/test_new.avi --output output/my_results
```

### Option 2: Sử dụng script bash tiện lợi

```bash
./run_full_analysis.sh
```

## 📂 Cấu trúc Output

```
output/test_new_full_crops/
├── detailed_analysis.txt           # Log tổng hợp
├── frame_0000_person_0.jpg         # Person crop từ frame 0
├── frame_0000_person_0.txt         # Thông số person crop
├── frame_0004_person_0.jpg         # Person 0 từ frame 4
├── frame_0004_person_0.txt         
├── frame_0004_person_1.jpg         # Person 1 từ frame 4
├── frame_0004_person_1.txt         
├── frame_0732_person_0_face_0.jpg  # Face crop (nếu có)
├── frame_0732_person_0_face_0.txt  # Thông số face crop
└── ...
```

## 📊 Ví dụ File Thông Số

### Person Crop (.txt)
```
Frame: 0
Số đối tượng: 1 person
Thời gian xử lý (person detection): 17.83 ms
Inference time (person): 17.83 ms
Accuracy (person): 1.0000
Kích thước ảnh crop: 450x900
Kích thước frame gốc: 1920x1080
FPS (tại thời điểm này): 56.02
```

### Face Crop (.txt)
```
Frame: 732
Số đối tượng: 1 face
Thời gian xử lý (face detection): 11.86 ms
Inference time (face): 11.86 ms
Accuracy (face): 0.9845
Kích thước ảnh crop: 120x120
Kích thước person crop: 450x900
Kích thước frame gốc: 1920x1080
FPS (tại thời điểm này): 15.18
```

## 🔍 Kiểm Tra Kết Quả

### Đếm số file đã tạo:
```bash
ls output/test_new_full_crops/*.jpg | wc -l
ls output/test_new_full_crops/*.txt | grep -v detailed | wc -l
```

### Xem file thông số mẫu:
```bash
cat output/test_new_full_crops/frame_0000_person_0.txt
```

### Tìm tất cả face crops:
```bash
ls output/test_new_full_crops/*face*.jpg
```

## ⚙️ Tham Số Script

```bash
python run_detailed_analysis.py --help

Options:
  --input, -i   : Đường dẫn đến video (bắt buộc)
  --output, -o  : Thư mục output (mặc định: output/analysis)
```

## 📈 Hiệu Năng

- **Video**: test_new.avi (836 frames, 1920x1080, 14.99 FPS)
- **Thời gian xử lý**: ~67 giây
- **FPS trung bình**: 15.31 FPS
- **Accuracy trung bình**: 59.41%
- **Thời gian xử lý/frame**: ~75 ms

## 💡 Lưu Ý

- Script tự động tạo thư mục output nếu chưa tồn tại
- Mỗi person và face được đánh số riêng biệt
- File txt luôn đi kèm với ảnh crop tương ứng
- Face chỉ được phát hiện trong person crops (không phải toàn bộ frame)
