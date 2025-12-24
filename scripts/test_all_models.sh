#!/bin/bash
#
# Test All Models Script
# Automatically tests all available models in the models/ directory
# Generates comprehensive reports and logs
#
# Usage: bash scripts/test_all_models.sh [input_file]
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"
OUTPUT_BASE="$PROJECT_DIR/output"
LOGS_DIR="$PROJECT_DIR/logs"
REPORTS_DIR="$PROJECT_DIR/docs/reports"

# Default input file
INPUT_FILE="${1:-$PROJECT_DIR/input/test.png}"

# Timestamp for this test run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_ID="test_run_$TIMESTAMP"

# Create directories
mkdir -p "$OUTPUT_BASE"
mkdir -p "$LOGS_DIR"
mkdir -p "$REPORTS_DIR"

# Log file for this run
MASTER_LOG="$LOGS_DIR/${RUN_ID}_master.log"
RESULTS_CSV="$REPORTS_DIR/${RUN_ID}_results.csv"
SUMMARY_REPORT="$REPORTS_DIR/${RUN_ID}_summary.md"

# Initialize log and CSV
echo "===========================================================" | tee "$MASTER_LOG"
echo "Model Testing Suite - Started at $(date)" | tee -a "$MASTER_LOG"
echo "===========================================================" | tee -a "$MASTER_LOG"
echo "Input file: $INPUT_FILE" | tee -a "$MASTER_LOG"
echo "Output directory: $OUTPUT_BASE" | tee -a "$MASTER_LOG"
echo "Run ID: $RUN_ID" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# CSV Header
echo "Model,Type,Size_MB,Pipeline,Processing_Time_s,FPS,Persons_Detected,Faces_Detected,Status,Notes" > "$RESULTS_CSV"

# Verify input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}Error: Input file not found: $INPUT_FILE${NC}" | tee -a "$MASTER_LOG"
    echo "Usage: $0 [input_file]" | tee -a "$MASTER_LOG"
    exit 1
fi

echo -e "${GREEN}✓ Input file found: $INPUT_FILE${NC}" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# Function to get file size in MB
get_size_mb() {
    local file="$1"
    if [ -f "$file" ]; then
        size_bytes=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "0")
        echo "scale=2; $size_bytes / 1048576" | bc
    else
        echo "0"
    fi
}

# Function to test a model
test_model() {
    local model_name="$1"
    local model_path="$2"
    local model_type="$3"  # "person" or "face"
    local pipeline_mode="$4"  # "sequential" or "parallel"
    
    echo -e "${BLUE}Testing: $model_name ($model_type, $pipeline_mode)${NC}" | tee -a "$MASTER_LOG"
    
    # Create output directory for this model
    local output_dir="$OUTPUT_BASE/${RUN_ID}/${model_name}_${pipeline_mode}"
    mkdir -p "$output_dir"
    
    # Get model size
    local model_size=$(get_size_mb "$model_path")
    
    # Build command
    local cmd="python $PROJECT_DIR/main.py --input \"$INPUT_FILE\" --output-dir \"$output_dir\" --pipeline $pipeline_mode"
    
    if [ "$model_type" == "person" ]; then
        cmd="$cmd --person-model \"$model_path\""
    else
        cmd="$cmd --face-model \"$model_path\""
    fi
    
    # Log file for this test
    local test_log="$LOGS_DIR/${RUN_ID}_${model_name}_${pipeline_mode}.log"
    
    # Run test
    local start_time=$(date +%s)
    local status="PASS"
    local notes=""
    local processing_time=0
    local fps=0
    local persons_detected=0
    local faces_detected=0
    
    echo "Command: $cmd" >> "$test_log"
    echo "Started at: $(date)" >> "$test_log"
    echo "---" >> "$test_log"
    
    if eval "$cmd" >> "$test_log" 2>&1; then
        local end_time=$(date +%s)
        processing_time=$((end_time - start_time))
        
        # Parse output for metrics
        if [ -f "$test_log" ]; then
            persons_detected=$(grep -oP "Detected \K[0-9]+ person" "$test_log" | head -1 | grep -oP "[0-9]+" || echo "0")
            faces_detected=$(grep -oP "and \K[0-9]+ face" "$test_log" | head -1 | grep -oP "[0-9]+" || echo "0")
            fps=$(grep -oP "Avg FPS: \K[0-9]+\.[0-9]+" "$test_log" | head -1 || echo "0")
        fi
        
        if [ "$processing_time" -gt 0 ]; then
            fps=$(echo "scale=2; 1 / $processing_time" | bc)
        fi
        
        echo -e "${GREEN}✓ PASSED${NC} - Time: ${processing_time}s, FPS: $fps" | tee -a "$MASTER_LOG"
        notes="Success"
    else
        local end_time=$(date +%s)
        processing_time=$((end_time - start_time))
        status="FAIL"
        notes="Error during processing. Check log: $test_log"
        echo -e "${RED}✗ FAILED${NC} - Check log: $test_log" | tee -a "$MASTER_LOG"
    fi
    
    # Append to CSV
    echo "$model_name,$model_type,$model_size,$pipeline_mode,$processing_time,$fps,$persons_detected,$faces_detected,$status,\"$notes\"" >> "$RESULTS_CSV"
    
    echo "---" >> "$test_log"
    echo "Completed at: $(date)" >> "$test_log"
    echo "Status: $status" >> "$test_log"
    echo "" | tee -a "$MASTER_LOG"
}

# Find and test all models
echo -e "${YELLOW}Scanning for models...${NC}" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# Test Person Detection Models
echo -e "${YELLOW}=== Testing Person Detection Models ===${NC}" | tee -a "$MASTER_LOG"

# NanoDet
if [ -f "$MODELS_DIR/NanoDet/object_detection_nanodet_2022nov.onnx" ]; then
    test_model "NanoDet-FP32" "$MODELS_DIR/NanoDet/object_detection_nanodet_2022nov.onnx" "person" "sequential"
    test_model "NanoDet-FP32" "$MODELS_DIR/NanoDet/object_detection_nanodet_2022nov.onnx" "person" "parallel"
fi

if [ -f "$MODELS_DIR/NanoDet/object_detection_nanodet_2022nov_int8.onnx" ]; then
    test_model "NanoDet-INT8" "$MODELS_DIR/NanoDet/object_detection_nanodet_2022nov_int8.onnx" "person" "sequential"
    test_model "NanoDet-INT8" "$MODELS_DIR/NanoDet/object_detection_nanodet_2022nov_int8.onnx" "person" "parallel"
fi

# NanoDet-Plus
if [ -f "$MODELS_DIR/NanoDet-Plus/nanodet-plus-m_320.onnx" ]; then
    test_model "NanoDet-Plus-320" "$MODELS_DIR/NanoDet-Plus/nanodet-plus-m_320.onnx" "person" "sequential"
fi

if [ -f "$MODELS_DIR/NanoDet-Plus/nanodet-plus-m_416.onnx" ]; then
    test_model "NanoDet-Plus-416" "$MODELS_DIR/NanoDet-Plus/nanodet-plus-m_416.onnx" "person" "sequential"
fi

# RF-DETR-Nano
if [ -f "$MODELS_DIR/RF-DETR-Nano/model.onnx" ]; then
    test_model "RF-DETR-Nano-FP32" "$MODELS_DIR/RF-DETR-Nano/model.onnx" "person" "sequential"
fi

if [ -f "$MODELS_DIR/RF-DETR-Nano/model_fp16.onnx" ]; then
    test_model "RF-DETR-Nano-FP16" "$MODELS_DIR/RF-DETR-Nano/model_fp16.onnx" "person" "sequential"
fi

if [ -f "$MODELS_DIR/RF-DETR-Nano/model_int8.onnx" ]; then
    test_model "RF-DETR-Nano-INT8" "$MODELS_DIR/RF-DETR-Nano/model_int8.onnx" "person" "sequential"
fi

if [ -f "$MODELS_DIR/RF-DETR-Nano/model_quantized.onnx" ]; then
    test_model "RF-DETR-Nano-Quantized" "$MODELS_DIR/RF-DETR-Nano/model_quantized.onnx" "person" "sequential"
fi

# EfficientDet-Lite
for lite_model in "$MODELS_DIR"/EfficientDet-Lite*/*.tflite; do
    if [ -f "$lite_model" ]; then
        model_name=$(basename "$(dirname "$lite_model")")_$(basename "$lite_model" .tflite)
        # Note: TFLite models need conversion, skip for now
        echo "Skipping TFLite model: $model_name (needs conversion)" | tee -a "$MASTER_LOG"
    fi
done

echo "" | tee -a "$MASTER_LOG"

# Test Face Detection Models
echo -e "${YELLOW}=== Testing Face Detection Models ===${NC}" | tee -a "$MASTER_LOG"

# YuNet
if [ -f "$MODELS_DIR/YuNet/face_detection_yunet_2023mar.onnx" ]; then
    test_model "YuNet-FP32" "$MODELS_DIR/YuNet/face_detection_yunet_2023mar.onnx" "face" "sequential"
    test_model "YuNet-FP32" "$MODELS_DIR/YuNet/face_detection_yunet_2023mar.onnx" "face" "parallel"
fi

if [ -f "$MODELS_DIR/YuNet/face_detection_yunet_2023mar_int8.onnx" ]; then
    test_model "YuNet-INT8" "$MODELS_DIR/YuNet/face_detection_yunet_2023mar_int8.onnx" "face" "sequential"
    test_model "YuNet-INT8" "$MODELS_DIR/YuNet/face_detection_yunet_2023mar_int8.onnx" "face" "parallel"
fi

# YOLOv8-Face
if [ -f "$MODELS_DIR/YOLOv8-Face/yolov8n-face.onnx" ]; then
    test_model "YOLOv8-Face" "$MODELS_DIR/YOLOv8-Face/yolov8n-face.onnx" "face" "sequential"
    test_model "YOLOv8-Face" "$MODELS_DIR/YOLOv8-Face/yolov8n-face.onnx" "face" "parallel"
fi

# UltraFace
if [ -f "$MODELS_DIR/UltraFace/ultraface_rfb_320.onnx" ]; then
    test_model "UltraFace-320" "$MODELS_DIR/UltraFace/ultraface_rfb_320.onnx" "face" "sequential"
fi

if [ -f "$MODELS_DIR/UltraFace/ultraface_rfb_640.onnx" ]; then
    test_model "UltraFace-640" "$MODELS_DIR/UltraFace/ultraface_rfb_640.onnx" "face" "sequential"
fi

# MediaPipe BlazeFace
if [ -f "$MODELS_DIR/MediaPipe-Face/blaze_face_short_range.tflite" ]; then
    echo "Skipping TFLite model: MediaPipe-BlazeFace (needs conversion)" | tee -a "$MASTER_LOG"
fi

echo "" | tee -a "$MASTER_LOG"

# Generate Summary Report
echo -e "${YELLOW}Generating summary report...${NC}" | tee -a "$MASTER_LOG"

cat > "$SUMMARY_REPORT" << EOF
# Model Testing Summary Report

**Test Run ID:** $RUN_ID  
**Date:** $(date)  
**Input File:** $INPUT_FILE  

---

## Test Results

EOF

# Parse CSV and generate markdown table
echo "| Model | Type | Size (MB) | Pipeline | Time (s) | FPS | Persons | Faces | Status |" >> "$SUMMARY_REPORT"
echo "|-------|------|-----------|----------|----------|-----|---------|-------|--------|" >> "$SUMMARY_REPORT"

tail -n +2 "$RESULTS_CSV" | while IFS=',' read -r model type size pipeline time fps persons faces status notes; do
    # Clean quotes from notes
    notes=$(echo "$notes" | tr -d '"')
    
    # Add emoji based on status
    if [ "$status" == "PASS" ]; then
        status_icon="✅ PASS"
    else
        status_icon="❌ FAIL"
    fi
    
    echo "| $model | $type | $size | $pipeline | $time | $fps | $persons | $faces | $status_icon |" >> "$SUMMARY_REPORT"
done

# Add statistics
cat >> "$SUMMARY_REPORT" << EOF

---

## Statistics

EOF

# Count pass/fail
total_tests=$(tail -n +2 "$RESULTS_CSV" | wc -l)
passed_tests=$(tail -n +2 "$RESULTS_CSV" | grep ",PASS," | wc -l)
failed_tests=$(tail -n +2 "$RESULTS_CSV" | grep ",FAIL," | wc -l)

cat >> "$SUMMARY_REPORT" << EOF
- **Total Tests:** $total_tests
- **Passed:** $passed_tests
- **Failed:** $failed_tests
- **Success Rate:** $(echo "scale=2; $passed_tests * 100 / $total_tests" | bc)%

---

## Performance Analysis

### Fastest Models (by FPS)
EOF

# Sort by FPS and get top 5
echo "\`\`\`" >> "$SUMMARY_REPORT"
tail -n +2 "$RESULTS_CSV" | grep ",PASS," | sort -t',' -k6 -rn | head -5 | \
    awk -F',' '{printf "%-30s | FPS: %6s | Time: %5ss\n", $1, $6, $5}' >> "$SUMMARY_REPORT"
echo "\`\`\`" >> "$SUMMARY_REPORT"

cat >> "$SUMMARY_REPORT" << EOF

### Smallest Models (by size)
\`\`\`
EOF

tail -n +2 "$RESULTS_CSV" | grep ",PASS," | sort -t',' -k3 -n | head -5 | \
    awk -F',' '{printf "%-30s | Size: %7s MB\n", $1, $3}' >> "$SUMMARY_REPORT"
echo "\`\`\`" >> "$SUMMARY_REPORT"

cat >> "$SUMMARY_REPORT" << EOF

### Most Accurate Models (by detection count)
\`\`\`
EOF

tail -n +2 "$RESULTS_CSV" | grep ",PASS," | \
    awk -F',' '{total=$7+$8; print $1","total}' | sort -t',' -k2 -rn | head -5 | \
    awk -F',' '{printf "%-30s | Detections: %s\n", $1, $2}' >> "$SUMMARY_REPORT"
echo "\`\`\`" >> "$SUMMARY_REPORT"

cat >> "$SUMMARY_REPORT" << EOF

---

## Recommendations

### For Speed (Real-time processing)
1. YuNet-INT8 (Face Detection)
2. NanoDet-INT8 (Person Detection)
3. Use Parallel Pipeline

### For Accuracy
1. YOLOv8-Face (Face Detection)
2. RF-DETR-Nano-FP32 (Person Detection)
3. Use Sequential Pipeline

### For Low Memory
1. UltraFace-320 (Face Detection)
2. NanoDet-INT8 (Person Detection)
3. Reduce input resolution

---

## Files Generated

- **Master Log:** \`$MASTER_LOG\`
- **Results CSV:** \`$RESULTS_CSV\`
- **Test Outputs:** \`$OUTPUT_BASE/$RUN_ID/\`
- **Individual Logs:** \`$LOGS_DIR/${RUN_ID}_*.log\`

---

## Next Steps

1. Review individual test logs for detailed error messages
2. Compare output images in \`$OUTPUT_BASE/$RUN_ID/\`
3. Select best models based on your requirements
4. Update \`src/config.py\` with chosen models

EOF

echo -e "${GREEN}✓ Summary report generated: $SUMMARY_REPORT${NC}" | tee -a "$MASTER_LOG"

# Final summary to console
echo "" | tee -a "$MASTER_LOG"
echo "===========================================================" | tee -a "$MASTER_LOG"
echo "Test Suite Completed at $(date)" | tee -a "$MASTER_LOG"
echo "===========================================================" | tee -a "$MASTER_LOG"
echo -e "${GREEN}Total Tests: $total_tests${NC}" | tee -a "$MASTER_LOG"
echo -e "${GREEN}Passed: $passed_tests${NC}" | tee -a "$MASTER_LOG"
echo -e "${RED}Failed: $failed_tests${NC}" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "Results saved to:" | tee -a "$MASTER_LOG"
echo "  - Summary: $SUMMARY_REPORT" | tee -a "$MASTER_LOG"
echo "  - CSV: $RESULTS_CSV" | tee -a "$MASTER_LOG"
echo "  - Master Log: $MASTER_LOG" | tee -a "$MASTER_LOG"
echo "  - Output: $OUTPUT_BASE/$RUN_ID/" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo -e "${BLUE}View report: cat $SUMMARY_REPORT${NC}"
echo -e "${BLUE}View CSV: cat $RESULTS_CSV${NC}"
echo ""

exit 0
