# Documentation Index

Welcome to the Human & Face Detection System documentation!

## 📚 Available Documentation

### User Documentation
- **[User Guide](USER_GUIDE.md)** - Complete guide for using the system
  - Installation instructions
  - Basic and advanced usage
  - Model management
  - Troubleshooting
  - Examples and best practices

### Technical Documentation
- **[Pipeline Architecture](PIPELINE_ARCHITECTURE.md)** - System architecture documentation
  - System overview
  - Pipeline modes (Sequential & Parallel)
  - Component architecture
  - Data flow diagrams
  - Performance optimization
  - Configuration details

### Reports
Reports are automatically generated when you run tests. They include:
- Test summary reports
- Performance analysis
- Sequence diagrams
- CSV results

Location: `reports/` (generated after running tests)

---

## 🚀 Quick Start

### 1. First Time Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Download models
python scripts/download_models.py
```

### 2. Basic Usage
```bash
# Process an image
python main.py --input input/photo.jpg

# Process a video
python main.py --input input/video.mp4
```

### 3. Test All Models
```bash
# Run comprehensive model testing
bash scripts/run_complete_test.sh input/test.png
```

This will:
- ✅ Test all available models
- ✅ Generate performance reports
- ✅ Create sequence diagrams
- ✅ Analyze logs automatically

---

## 📖 Documentation Structure

```
docs/
├── README.md                          # This file
├── USER_GUIDE.md                      # User guide
├── PIPELINE_ARCHITECTURE.md           # Technical architecture
└── reports/                           # Generated reports
    ├── test_run_YYYYMMDD_HHMMSS_summary.md
    ├── test_run_YYYYMMDD_HHMMSS_performance_analysis.md
    ├── test_run_YYYYMMDD_HHMMSS_sequence_diagram.md
    └── test_run_YYYYMMDD_HHMMSS_results.csv
```

---

## 🎯 Document Purpose

| Document | Purpose | Audience |
|----------|---------|----------|
| [USER_GUIDE.md](USER_GUIDE.md) | How to use the system | End users, Operators |
| [PIPELINE_ARCHITECTURE.md](PIPELINE_ARCHITECTURE.md) | How the system works | Developers, Architects |
| Test Reports | Performance analysis | Developers, QA |
| Sequence Diagrams | Process flows | All technical staff |

---

## 🔧 Testing & Analysis Tools

### Run Complete Test Suite
```bash
# Test all models with one command
bash scripts/run_complete_test.sh input/test.png
```

**What it does:**
1. Creates necessary directories
2. Tests all models (person + face detection)
3. Tests both pipeline modes (sequential + parallel)
4. Generates comprehensive reports
5. Creates sequence diagrams
6. Analyzes performance metrics

### Manual Testing
```bash
# Test specific models
bash scripts/test_all_models.sh input/photo.jpg

# Analyze logs separately
python scripts/analyze_logs.py
```

### View Results
```bash
# View latest summary
cat docs/reports/test_run_*_summary.md | tail -n 100

# View performance analysis
cat docs/reports/test_run_*_performance_analysis.md | tail -n 100

# View CSV results
cat docs/reports/test_run_*_results.csv
```

---

## 📊 Understanding Reports

### Summary Report
Contains:
- Test results table (all models tested)
- Statistics (pass/fail rates)
- Top performers by speed, size, accuracy
- Recommendations

### Performance Analysis Report
Contains:
- Detailed performance metrics
- Model comparisons (person vs face)
- Pipeline comparisons (sequential vs parallel)
- Optimization recommendations
- Failed tests analysis

### Sequence Diagram
Contains:
- Visual flow diagrams (Mermaid format)
- Sequential pipeline flow
- Parallel pipeline flow
- Detailed processing steps
- Model loading flow
- Metrics collection flow

### CSV Results
Raw data including:
- Model name, type, size
- Pipeline mode
- Processing time, FPS
- Detection counts
- Status (PASS/FAIL)

---

## 🎓 Learning Path

### For New Users
1. Start with [USER_GUIDE.md](USER_GUIDE.md) - Quick Start section
2. Try basic examples
3. Read troubleshooting section if issues arise

### For Developers
1. Read [PIPELINE_ARCHITECTURE.md](PIPELINE_ARCHITECTURE.md)
2. Understand component architecture
3. Review data flow diagrams
4. Run tests and analyze reports

### For System Integration
1. Understand pipeline modes
2. Review performance metrics
3. Select appropriate models for your use case
4. Configure based on hardware constraints

---

## 🔍 Model Selection Guide

### Quick Reference

| Use Case | Person Model | Face Model | Pipeline |
|----------|-------------|------------|----------|
| Real-time (>5 FPS) | NanoDet-INT8 | YuNet-INT8 | Parallel |
| High Accuracy | RF-DETR-FP32 | YOLOv8-Face | Sequential |
| Low Memory (<1GB) | NanoDet-INT8 | YuNet-INT8 | Sequential |
| Balanced | NanoDet-Plus | UltraFace-640 | Sequential |

See [USER_GUIDE.md - Model Management](USER_GUIDE.md#model-management) for details.

---

## 🛠️ Customization

### Configuration
Edit `src/config.py` to change:
- Default models
- Pipeline mode
- Thresholds
- Input size limits
- Thread count
- Output settings

### Adding New Models
1. Place ONNX model in `models/` folder
2. Test with:
   ```bash
   python main.py --input input/test.jpg \
       --person-model models/your_model.onnx
   ```
3. Run full test suite to benchmark

### Custom Pipeline
See [PIPELINE_ARCHITECTURE.md - Extension Points](PIPELINE_ARCHITECTURE.md#extension-points)

---

## 📝 Contributing to Documentation

### Adding New Documentation
1. Create markdown file in `docs/`
2. Add link to this README
3. Follow existing structure and style

### Updating Existing Docs
1. Edit relevant markdown file
2. Update table of contents if needed
3. Test all code examples

### Report Issues
Found errors or unclear documentation? Please report:
- GitHub Issues: [Link to issues]
- Include: document name, section, description

---

## 🔗 External Resources

### Model Sources
- **OpenCV Zoo:** https://github.com/opencv/opencv_zoo
- **Hugging Face:** https://huggingface.co/
- **ONNX Model Zoo:** https://github.com/onnx/models

### ONNX Runtime
- **Documentation:** https://onnxruntime.ai/docs/
- **GitHub:** https://github.com/microsoft/onnxruntime

### OpenCV
- **Documentation:** https://docs.opencv.org/
- **Tutorials:** https://docs.opencv.org/master/d9/df8/tutorial_root.html

---

## 📧 Support

### Getting Help
1. Check [USER_GUIDE.md - Troubleshooting](USER_GUIDE.md#troubleshooting)
2. Review test reports for similar issues
3. Search GitHub issues
4. Create new issue with:
   - System info
   - Error message
   - Log files
   - Steps to reproduce

### Documentation Feedback
Help us improve! Suggest:
- Missing information
- Unclear explanations
- Additional examples
- Better organization

---

## 📅 Version History

- **v1.0** - Initial documentation
  - User Guide
  - Pipeline Architecture
  - Testing suite with automated reporting

---

## 🎉 Thank You!

Thank you for using the Human & Face Detection System. We hope this documentation helps you get the most out of the system!

**Happy Detecting!** 🎯

---

*Last Updated: December 24, 2025*
