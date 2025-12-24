#!/bin/bash
#
# Master Test Runner
# Runs complete testing suite: test models -> analyze logs -> generate reports
#
# Usage: bash scripts/run_complete_test.sh [input_file]
#

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Complete Model Testing & Analysis Suite               ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Input file
INPUT_FILE="${1:-$PROJECT_DIR/input/test.png}"

# Step 1: Create directories
echo -e "${YELLOW}Step 1: Creating directories...${NC}"
mkdir -p "$PROJECT_DIR/output"
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/docs/reports"
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Step 2: Verify input file
echo -e "${YELLOW}Step 2: Verifying input file...${NC}"
if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}Error: Input file not found: $INPUT_FILE${NC}"
    echo "Available test images:"
    ls -1 "$PROJECT_DIR/input/" 2>/dev/null || echo "No files in input/"
    exit 1
fi
echo -e "${GREEN}✓ Input file verified: $INPUT_FILE${NC}"
echo ""

# Step 3: Run model tests
echo -e "${YELLOW}Step 3: Running model tests...${NC}"
echo "This may take several minutes..."
echo ""

if bash "$SCRIPT_DIR/test_all_models.sh" "$INPUT_FILE"; then
    echo -e "${GREEN}✓ Model testing completed${NC}"
else
    echo -e "${RED}✗ Model testing failed${NC}"
    exit 1
fi
echo ""

# Step 4: Analyze logs and generate reports
echo -e "${YELLOW}Step 4: Analyzing logs and generating reports...${NC}"
if python "$SCRIPT_DIR/analyze_logs.py"; then
    echo -e "${GREEN}✓ Log analysis completed${NC}"
else
    echo -e "${RED}✗ Log analysis failed${NC}"
    exit 1
fi
echo ""

# Step 5: Generate final summary
echo -e "${YELLOW}Step 5: Generating final summary...${NC}"

# Find latest run
LATEST_RUN=$(ls -1t "$PROJECT_DIR/logs"/test_run_*_master.log 2>/dev/null | head -1 | xargs basename | sed 's/_master.log//')

if [ -z "$LATEST_RUN" ]; then
    echo -e "${RED}No test run found${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Latest test run: $LATEST_RUN${NC}"
echo ""

# Display summary
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    Test Results Summary                    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Parse CSV for quick summary
CSV_FILE="$PROJECT_DIR/docs/reports/${LATEST_RUN}_results.csv"
if [ -f "$CSV_FILE" ]; then
    TOTAL=$(tail -n +2 "$CSV_FILE" | wc -l)
    PASSED=$(tail -n +2 "$CSV_FILE" | grep ",PASS," | wc -l)
    FAILED=$(tail -n +2 "$CSV_FILE" | grep ",FAIL," | wc -l)
    
    echo -e "${GREEN}Total Tests: $TOTAL${NC}"
    echo -e "${GREEN}Passed: $PASSED${NC}"
    echo -e "${YELLOW}Failed: $FAILED${NC}"
    echo ""
    
    # Show top 3 fastest models
    echo -e "${BLUE}Top 3 Fastest Models:${NC}"
    tail -n +2 "$CSV_FILE" | grep ",PASS," | sort -t',' -k6 -rn | head -3 | \
        awk -F',' '{printf "  %d. %-30s | FPS: %6s | Pipeline: %s\n", NR, $1, $6, $4}'
    echo ""
fi

# Step 6: Display file locations
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    Generated Files                         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo "📊 Reports:"
echo "  - Summary: docs/reports/${LATEST_RUN}_summary.md"
echo "  - Performance Analysis: docs/reports/${LATEST_RUN}_performance_analysis.md"
echo "  - Sequence Diagrams: docs/reports/${LATEST_RUN}_sequence_diagram.md"
echo "  - CSV Results: docs/reports/${LATEST_RUN}_results.csv"
echo ""

echo "📝 Logs:"
echo "  - Master Log: logs/${LATEST_RUN}_master.log"
echo "  - Individual Logs: logs/${LATEST_RUN}_*.log"
echo ""

echo "🖼️  Output Images:"
echo "  - All outputs: output/${LATEST_RUN}/"
echo ""

# Step 7: Quick view commands
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    Quick View Commands                     ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo "View summary report:"
echo "  cat docs/reports/${LATEST_RUN}_summary.md"
echo ""

echo "View performance analysis:"
echo "  cat docs/reports/${LATEST_RUN}_performance_analysis.md"
echo ""

echo "View sequence diagrams:"
echo "  cat docs/reports/${LATEST_RUN}_sequence_diagram.md"
echo ""

echo "View CSV results:"
echo "  cat docs/reports/${LATEST_RUN}_results.csv"
echo ""

echo "Browse output images:"
echo "  ls -lh output/${LATEST_RUN}/"
echo ""

# Optional: Open reports if on desktop system
if command -v xdg-open &> /dev/null; then
    echo -e "${YELLOW}Open reports in browser? (y/n)${NC}"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        # Convert markdown to HTML if pandoc is available
        if command -v pandoc &> /dev/null; then
            echo "Converting reports to HTML..."
            pandoc "docs/reports/${LATEST_RUN}_summary.md" -o "docs/reports/${LATEST_RUN}_summary.html"
            pandoc "docs/reports/${LATEST_RUN}_performance_analysis.md" -o "docs/reports/${LATEST_RUN}_performance_analysis.html"
            xdg-open "docs/reports/${LATEST_RUN}_summary.html"
        else
            echo "Pandoc not installed. Install with: sudo apt install pandoc"
        fi
    fi
fi

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║          ✨ All Tests and Analysis Complete! ✨           ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

exit 0
