#!/bin/bash
# Quick Start Guide - One-line commands for common tasks

echo "╔════════════════════════════════════════════════════════════╗"
echo "║       Detected Human Faces - Quick Command Reference      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

cat << 'EOF'
🚀 QUICK START COMMANDS

1. Download Models (First Time Setup)
   $ python scripts/download_models.py

2. Process Single Image
   $ python main.py --input input/photo.jpg

3. Process Video
   $ python main.py --input input/video.mp4

4. Test All Models (Complete Test Suite)
   $ bash scripts/run_complete_test.sh input/test.png

5. View Latest Test Report
   $ cat docs/reports/test_run_*_summary.md | tail -n 100

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚡ OPTIMIZED CONFIGURATIONS

For Speed (Real-time):
$ python main.py --input input/video.mp4 \
    --person-model models/NanoDet/object_detection_nanodet_2022nov_int8.onnx \
    --face-model models/YuNet/face_detection_yunet_2023mar_int8.onnx \
    --pipeline parallel

For Accuracy:
$ python main.py --input input/photo.jpg \
    --person-model models/RF-DETR-Nano/model.onnx \
    --face-model models/YOLOv8-Face/yolov8n-face.onnx \
    --pipeline sequential \
    --person-threshold 0.6 \
    --face-threshold 0.6

For Low Memory:
$ python main.py --input input/photo.jpg \
    --person-model models/NanoDet/object_detection_nanodet_2022nov_int8.onnx \
    --face-model models/YuNet/face_detection_yunet_2023mar_int8.onnx

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 VIEW RESULTS

List Downloaded Models:
$ ls -lh models/*/

View Output Images:
$ ls -lh output/

View Logs:
$ ls -lh logs/

View Reports:
$ ls -lh docs/reports/

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📖 DOCUMENTATION

User Guide:
$ cat docs/USER_GUIDE.md

Pipeline Architecture:
$ cat docs/PIPELINE_ARCHITECTURE.md

Model Information:
$ cat models/DOWNLOADED_MODELS.md

Project Summary:
$ cat PROJECT_COMPLETION_SUMMARY.md

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🛠️ UTILITIES

Test Specific Model:
$ python main.py --input input/test.png \
    --person-model models/NanoDet-Plus/nanodet-plus-m_416.onnx

Analyze Logs:
$ python scripts/analyze_logs.py

Process Batch Images:
$ for img in input/*.jpg; do
    python main.py --input "$img"
done

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📁 PROJECT STRUCTURE

detected-human-faces/
├── src/              # Source code
├── scripts/          # Utility scripts
├── docs/             # Documentation
├── models/           # ONNX models (19 files)
├── input/            # Your images/videos
├── output/           # Detection results
└── logs/             # Performance logs

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 TIPS

- Use INT8 models for speed (e.g., nanodet_int8.onnx)
- Use Sequential pipeline for multiple persons
- Use Parallel pipeline for single person/speed
- Lower thresholds (0.3-0.4) to detect more objects
- Higher thresholds (0.6-0.7) for fewer, confident detections

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🆘 TROUBLESHOOTING

Out of Memory:
→ Use INT8 quantized models
→ Reduce MAX_INPUT_WIDTH in src/config.py

Too Slow:
→ Use --pipeline parallel
→ Use smaller models (NanoDet-INT8, YuNet-INT8)

No Detections:
→ Lower thresholds: --person-threshold 0.3
→ Try different models

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For detailed help: docs/USER_GUIDE.md
For architecture: docs/PIPELINE_ARCHITECTURE.md

Happy Detecting! 🎯
EOF
