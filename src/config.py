"""
Configuration file for human and face detection system
Optimized for Orange Pi RV 2 (4GB RAM)
"""

import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Project root
MODEL_DIR = os.path.join(BASE_DIR, "models")
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Pipeline modes
PIPELINE_PARALLEL = "parallel"  # Crop person and face in parallel
PIPELINE_SEQUENTIAL = "sequential"  # Crop person first, then face

# Default pipeline mode
DEFAULT_PIPELINE = PIPELINE_SEQUENTIAL

# Model paths (can be changed by user)
PERSON_MODEL_PATH = os.path.join(MODEL_DIR, "person_detection.onnx")
FACE_MODEL_PATH = os.path.join(MODEL_DIR, "face_detection.onnx")

# Detection thresholds
PERSON_CONFIDENCE_THRESHOLD = 0.5
FACE_CONFIDENCE_THRESHOLD = 0.5

# Image/Video processing settings
# Lower resolution for Orange Pi RV 2 optimization
MAX_INPUT_WIDTH = 640
MAX_INPUT_HEIGHT = 480

# Output settings
SAVE_CROPPED_IMAGES = True
SAVE_ANNOTATED_OUTPUT = True
SAVE_METRICS_TXT = True

# Performance optimization for Orange Pi
USE_THREADING = True
MAX_THREADS = 2  # Limited for 4GB RAM

# Supported formats
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp']
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv']
