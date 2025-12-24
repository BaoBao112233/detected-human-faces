#!/bin/bash
# Script to download remaining models that failed

echo "============================================"
echo "Downloading Remaining Models"
echo "============================================"

cd "$(dirname "$0")/models"

# SCRFD Models (Alternative source)
echo -e "\n[SCRFD] Downloading from alternative sources..."
mkdir -p SCRFD
cd SCRFD

# Download SCRFD 500M
if [ ! -f "scrfd_500m_bnkps.onnx" ]; then
    echo "Downloading SCRFD 500M..."
    wget -q --show-progress "https://github.com/opencv/opencv_zoo/raw/dev/models/face_detection_scrfd/face_detection_scrfd_500m_bnkps.onnx" \
         -O scrfd_500m_bnkps.onnx 2>/dev/null || \
    curl -L "https://drive.google.com/uc?export=download&id=1fZnGRUQW0dR7LRfFGlPOC1Kt5P9_wvZT" \
         -o scrfd_500m_bnkps.onnx 2>/dev/null || \
    echo "Failed to download SCRFD 500M. Please download manually from:"
    echo "https://github.com/deepinsight/insightface/tree/master/detection/scrfd"
fi

# Download SCRFD 2.5G
if [ ! -f "scrfd_2.5g_bnkps.onnx" ]; then
    echo "Downloading SCRFD 2.5G..."
    wget -q --show-progress "https://github.com/opencv/opencv_zoo/raw/dev/models/face_detection_scrfd/face_detection_scrfd_2.5g_bnkps.onnx" \
         -O scrfd_2.5g_bnkps.onnx 2>/dev/null || \
    echo "Failed to download SCRFD 2.5G. Please download manually."
fi

cd ..

# EdgeYOLO Models
echo -e "\n[EdgeYOLO] Downloading..."
mkdir -p EdgeYOLO
cd EdgeYOLO

if [ ! -f "edgeyolo_coco.onnx" ]; then
    echo "Downloading EdgeYOLO COCO..."
    # Try alternative sources
    wget -q --show-progress "https://github.com/LSH9832/edgeyolo/releases/download/v0.1.0/edgeyolo.onnx" \
         -O edgeyolo_coco.onnx 2>/dev/null || \
    echo "Failed. Manual download required from:"
    echo "https://github.com/LSH9832/edgeyolo"
fi

if [ ! -f "edgeyolo_tiny_coco.onnx" ]; then
    echo "Downloading EdgeYOLO Tiny..."
    wget -q --show-progress "https://github.com/LSH9832/edgeyolo/releases/download/v0.1.0/edgeyolo_tiny.onnx" \
         -O edgeyolo_tiny_coco.onnx 2>/dev/null || \
    echo "Failed. Manual download required."
fi

cd ..

# YOLOv8 Face (Alternative)
echo -e "\n[YOLOv8-Face] Trying alternative source..."
cd YOLOv8-Face

if [ ! -f "yolov8n-face.onnx" ]; then
    echo "Downloading YOLOv8n-Face..."
    # Try direct download from various sources
    wget -q --show-progress "https://github.com/hpc203/yolov8-face-landmarks-opencv-dnn/raw/main/yolov8n-face.onnx" \
         -O yolov8n-face.onnx 2>/dev/null || \
    echo "Already exists or failed to download."
fi

cd ../..

echo -e "\n============================================"
echo "Alternative Download Methods:"
echo "============================================"
echo ""
echo "For SCRFD models:"
echo "  git clone --depth 1 https://github.com/deepinsight/insightface.git"
echo "  cd insightface/python-package/insightface/model_zoo"
echo ""
echo "For EdgeYOLO models:"
echo "  git clone --depth 1 https://github.com/LSH9832/edgeyolo.git"
echo "  cd edgeyolo && bash scripts/download_onnx.sh"
echo ""
echo "Or download manually and place in models/ folder"
echo "============================================"

# Show summary
echo -e "\nChecking downloaded models..."
find models -name "*.onnx" -o -name "*.tflite" -o -name "*.tar" | wc -l | xargs echo "Total files:"
du -sh models/ 2>/dev/null | awk '{print "Total size: " $1}'

echo -e "\n✓ Done! Check models/ directory"
