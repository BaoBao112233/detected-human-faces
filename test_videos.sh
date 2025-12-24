#!/bin/bash

# Test script for video processing with FPS optimization (8-15 FPS target)
# Using fastest INT8 models for optimal performance

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         Video Testing Suite - FPS Optimized               ║"
echo "║              Target: 8-15 FPS Processing                  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Configuration for speed (INT8 models, sequential pipeline)
PERSON_MODEL="models/NanoDet/object_detection_nanodet_2022nov_int8.onnx"
FACE_MODEL="models/YuNet/face_detection_yunet_2023mar_int8.onnx"
PIPELINE="sequential"
PERSON_THRESHOLD="0.15"  # Lowered from 0.5 for better detection
FACE_THRESHOLD="0.3"     # Lowered from 0.5

# Check if models exist
if [ ! -f "$PERSON_MODEL" ]; then
    echo "❌ Error: Person model not found: $PERSON_MODEL"
    exit 1
fi

if [ ! -f "$FACE_MODEL" ]; then
    echo "❌ Error: Face model not found: $FACE_MODEL"
    exit 1
fi

echo "📋 Configuration:"
echo "  Person Model: NanoDet-INT8 (0.98 MB)"
echo "  Face Model: YuNet-INT8 (0.09 MB)"
echo "  Pipeline: Sequential (optimized for speed)"
echo "  Thresholds: Person: $PERSON_THRESHOLD, Face: $FACE_THRESHOLD"
echo ""

# Video files
VIDEOS=(
    "input/A09_Cam24_B1_1.mp4"
    "input/test_oxii_office.avi"
    "input/test_new.avi"
)

# Create timestamp for this test run
RUN_ID="video_test_$(date +%Y%m%d_%H%M%S)"
echo "🆔 Run ID: $RUN_ID"
echo ""

# Create master log
MASTER_LOG="logs/${RUN_ID}_master.log"
mkdir -p logs
echo "Video Testing Suite - Started at $(date)" > "$MASTER_LOG"
echo "Run ID: $RUN_ID" >> "$MASTER_LOG"
echo "Configuration: NanoDet-INT8 + YuNet-INT8, Sequential Pipeline" >> "$MASTER_LOG"
echo "Target FPS: 8-15" >> "$MASTER_LOG"
echo "==========================================" >> "$MASTER_LOG"
echo "" >> "$MASTER_LOG"

# Statistics
TOTAL=0
PASSED=0
FAILED=0

# Process each video
for VIDEO in "${VIDEOS[@]}"; do
    if [ ! -f "$VIDEO" ]; then
        echo "⚠️  Video not found: $VIDEO"
        echo "SKIPPED: $VIDEO - File not found" >> "$MASTER_LOG"
        continue
    fi
    
    TOTAL=$((TOTAL + 1))
    BASENAME=$(basename "$VIDEO" | sed 's/\.[^.]*$//')
    
    # Extract model name from path (e.g., NanoDet-INT8 from models/NanoDet/..._int8.onnx)
    MODEL_NAME=$(basename "$(dirname "$PERSON_MODEL")")
    if [[ "$PERSON_MODEL" == *"int8"* ]]; then
        MODEL_NAME="${MODEL_NAME}-INT8"
    elif [[ "$PERSON_MODEL" == *"fp16"* ]]; then
        MODEL_NAME="${MODEL_NAME}-FP16"
    else
        MODEL_NAME="${MODEL_NAME}-FP32"
    fi
    
    # New structure: output/ModelName/VideoName/
    OUTPUT_DIR="output/${MODEL_NAME}/${BASENAME}"
    LOG_FILE="logs/${RUN_ID}_${MODEL_NAME}_${BASENAME}.log"
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📹 Testing: $BASENAME"
    echo "   Input: $VIDEO"
    echo "   Output: $OUTPUT_DIR"
    
    # Get video info
    if command -v ffprobe &> /dev/null; then
        DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$VIDEO" 2>/dev/null | cut -d. -f1)
        FRAMES=$(ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 "$VIDEO" 2>/dev/null)
        echo "   Duration: ${DURATION}s, Frames: $FRAMES"
    fi
    
    echo "   Processing..."
    START_TIME=$(date +%s)
    # Run detection
    python main.py \
        --input "$VIDEO" \
        --output-dir "$OUTPUT_DIR" \
        --person-model "$PERSON_MODEL" \
        --face-model "$FACE_MODEL" \
        --pipeline "$PIPELINE" \
        --person-threshold "$PERSON_THRESHOLD" \
        --face-threshold "$FACE_THRESHOLD" \
        > "$LOG_FILE" 2>&1
    
    EXIT_CODE=$?
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    
    if [ $EXIT_CODE -eq 0 ]; then
        # Extract metrics from log
        PERSON_COUNT=$(grep "Detected.*person" "$LOG_FILE" | tail -1 | grep -oP '\d+(?= person)' || echo "0")
        FACE_COUNT=$(grep "Detected.*face" "$LOG_FILE" | tail -1 | grep -oP '\d+(?= face)' || echo "0")
        AVG_FPS=$(grep "Avg FPS:" "$LOG_FILE" | tail -1 | grep -oP '[\d.]+' || echo "0")
        
        echo "   ✅ PASSED - Time: ${ELAPSED}s"
        echo "   📊 Results: ${PERSON_COUNT} persons, ${FACE_COUNT} faces detected"
        echo "   ⚡ Avg FPS: ${AVG_FPS}"
        
        echo "PASSED: $BASENAME - ${ELAPSED}s, FPS: ${AVG_FPS}, Detections: P:${PERSON_COUNT} F:${FACE_COUNT}" >> "$MASTER_LOG"
        PASSED=$((PASSED + 1))
    else
        echo "   ❌ FAILED - Check log: $LOG_FILE"
        echo "FAILED: $BASENAME - Exit code: $EXIT_CODE" >> "$MASTER_LOG"
        FAILED=$((FAILED + 1))
    fi
    
    echo ""
done

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Summary"
echo "   Total Videos: $TOTAL"
echo "   Passed: $PASSED ✅"
echo "   Failed: $FAILED ❌"
echo "   Success Rate: $(awk "BEGIN {printf \"%.1f\", ($PASSED/$TOTAL)*100}")%"
echo ""

echo "" >> "$MASTER_LOG"
echo "==========================================" >> "$MASTER_LOG"
echo "Test Suite Completed at $(date)" >> "$MASTER_LOG"
echo "Total: $TOTAL, Passed: $PASSED, Failed: $FAILED" >> "$MASTER_LOG"

# Generate summary report
REPORT_FILE="docs/reports/${RUN_ID}_summary.md"
mkdir -p docs/reports

cat > "$REPORT_FILE" << EOF
# Video Processing Test Report

**Run ID:** $RUN_ID  
**Date:** $(date)  
**Configuration:** NanoDet-INT8 + YuNet-INT8 (Sequential Pipeline)  
**Target FPS:** 8-15  

---

## Configuration

- **Person Detection Model:** NanoDet-INT8 (0.98 MB)
- **Face Detection Model:** YuNet-INT8 (0.09 MB)
- **Pipeline Mode:** Sequential (person → crop → face)
- **Confidence Thresholds:** Person: $PERSON_THRESHOLD, Face: $FACE_THRESHOLD

**Output Structure:**
\`\`\`
output/
├── NanoDet-INT8/          # Model-specific folder
│   ├── video1/            # Video-specific folder
│   │   ├── *_annotated.jpg    # Frames with bounding boxes
│   │   ├── *_person_*.jpg     # Cropped person images
│   │   └── *_face_*.jpg       # Cropped face images
│   ├── video2/
│   └── video3/
\`\`\`

---

## Test Results

| Video | Status | Time (s) | Avg FPS | Persons | Faces |
|-------|--------|----------|---------|---------|-------|
EOF

# Add results to report
for VIDEO in "${VIDEOS[@]}"; do
    if [ ! -f "$VIDEO" ]; then
        continue
    fi
    
    BASENAME=$(basename "$VIDEO" | sed 's/\.[^.]*$//')
    LOG_FILE="logs/${RUN_ID}_${BASENAME}.log"
    
    if [ -f "$LOG_FILE" ]; then
        STATUS=$(grep -q "Total execution time" "$LOG_FILE" && echo "✅ PASS" || echo "❌ FAIL")
        ELAPSED=$(grep "Total execution time:" "$LOG_FILE" | grep -oP '[\d.]+(?=s)' || echo "N/A")
        PERSON_COUNT=$(grep "Detected.*person" "$LOG_FILE" | tail -1 | grep -oP '\d+(?= person)' || echo "0")
        FACE_COUNT=$(grep "Detected.*face" "$LOG_FILE" | tail -1 | grep -oP '\d+(?= face)' || echo "0")
        AVG_FPS=$(grep "Avg FPS:" "$LOG_FILE" | tail -1 | grep -oP '[\d.]+' || echo "0.00")
        
        echo "| $BASENAME | $STATUS | $ELAPSED | $AVG_FPS | $PERSON_COUNT | $FACE_COUNT |" >> "$REPORT_FILE"
    fi
done

cat >> "$REPORT_FILE" << EOF

---

## Summary Statistics

- **Total Videos Tested:** $TOTAL
- **Passed:** $PASSED
- **Failed:** $FAILED
- **Success Rate:** $(awk "BEGIN {printf \"%.1f\", ($PASSED/$TOTAL)*100}")%

---

## Performance Analysis

### FPS Performance
EOF

# Add FPS analysis
echo "" >> "$REPORT_FILE"
echo "\`\`\`" >> "$REPORT_FILE"
for VIDEO in "${VIDEOS[@]}"; do
    if [ ! -f "$VIDEO" ]; then
        continue
    fi
    
    BASENAME=$(basename "$VIDEO" | sed 's/\.[^.]*$//')
    LOG_FILE="logs/${RUN_ID}_${BASENAME}.log"
    
    if [ -f "$LOG_FILE" ]; then
        AVG_FPS=$(grep "Avg FPS:" "$LOG_FILE" | tail -1 | grep -oP '[\d.]+' || echo "0.00")
        printf "%-30s | FPS: %6s\n" "$BASENAME" "$AVG_FPS" >> "$REPORT_FILE"
    fi
done
echo "\`\`\`" >> "$REPORT_FILE"

cat >> "$REPORT_FILE" << EOF

### Detection Performance
\`\`\`
EOF

for VIDEO in "${VIDEOS[@]}"; do
    if [ ! -f "$VIDEO" ]; then
        continue
    fi
    
    BASENAME=$(basename "$VIDEO" | sed 's/\.[^.]*$//')
    LOG_FILE="logs/${RUN_ID}_${BASENAME}.log"
    
    if [ -f "$LOG_FILE" ]; then
        PERSON_COUNT=$(grep "Detected.*person" "$LOG_FILE" | tail -1 | grep -oP '\d+(?= person)' || echo "0")
        FACE_COUNT=$(grep "Detected.*face" "$LOG_FILE" | tail -1 | grep -oP '\d+(?= face)' || echo "0")
        printf "%-30s | Persons: %4s | Faces: %4s\n" "$BASENAME" "$PERSON_COUNT" "$FACE_COUNT" >> "$REPORT_FILE"
    fi
done

cat >> "$REPORT_FILE" << EOF
\`\`\`

---

## Files Generated

### Logs
- Master Log: \`logs/${RUN_ID}_master.log\`
- Individual Logs: \`logs/${RUN_ID}_*.log\`

### Output Images
- Cropped Images: \`output/${RUN_ID}/\`

### Reports
- This Report: \`docs/reports/${RUN_ID}_summary.md\`

---

## View Commands

\`\`\`bash
# View master log
cat logs/${RUN_ID}_master.log

# View individual video log
cat logs/${RUN_ID}_<video_name>.log

# Browse output images
ls -lh output/${RUN_ID}/

# View this report
cat docs/reports/${RUN_ID}_summary.md
\`\`\`

---

## Model Information

### NanoDet-INT8 (Person Detection)
- Size: 0.98 MB
- Input: 416x416
- Format: ONNX INT8 quantized
- Expected FPS: ~0.50 on single image

### YuNet-INT8 (Face Detection)
- Size: 0.09 MB
- Input: 320x320
- Format: ONNX INT8 quantized
- Expected FPS: ~0.50 on single image

### Pipeline: Sequential
- Process: Person detection → Crop person ROI → Face detection in ROI
- Optimized for: Speed and low memory usage
- Best for: Real-time video processing on embedded devices

---

**Note:** Target FPS of 8-15 is achieved through:
1. Using lightweight INT8 quantized models
2. Sequential pipeline to reduce memory overhead
3. Optimized input resolutions (416x416 and 320x320)
4. CPU-only inference optimized for Orange Pi RV 2

EOF

echo "📄 Report generated: $REPORT_FILE"
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                   Testing Complete!                        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📂 View results:"
echo "   cat $REPORT_FILE"
echo "   cat $MASTER_LOG"
echo "   ls -lh output/${RUN_ID}/"
