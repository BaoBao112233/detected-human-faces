"""
Detector classes for person and face detection
Supports ONNX models for optimized inference on Orange Pi
"""

import cv2
import numpy as np
import os
from typing import List, Tuple, Optional


class Detection:
    """Class to store detection results"""
    def __init__(self, bbox: Tuple[int, int, int, int], confidence: float, class_id: int = 0):
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.confidence = confidence
        self.class_id = class_id
    
    def get_crop(self, image: np.ndarray) -> np.ndarray:
        """Crop the detection from the image"""
        x1, y1, x2, y2 = self.bbox
        return image[y1:y2, x1:x2]


class BaseDetector:
    """Base class for detectors"""
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the detection model"""
        if not os.path.exists(self.model_path):
            print(f"Warning: Model not found at {self.model_path}")
            print("Using fallback cascade/HOG detector")
            return
        
        try:
            # Try to load ONNX model
            import onnxruntime as ort
            self.model = ort.InferenceSession(
                self.model_path,
                providers=['CPUExecutionProvider']  # CPU only for Orange Pi
            )
            print(f"Loaded ONNX model: {self.model_path}")
        except Exception as e:
            print(f"Failed to load ONNX model: {e}")
            print("Using fallback detector")
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """Detect objects in image - to be implemented by subclasses"""
        raise NotImplementedError


class PersonDetector(BaseDetector):
    """Person detector using ONNX model or fallback HOG"""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        super().__init__(model_path, confidence_threshold)
        # Fallback HOG detector for person detection
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """Detect persons in image"""
        if self.model is not None:
            return self._detect_onnx(image)
        else:
            return self._detect_hog(image)
    
    def _detect_onnx(self, image: np.ndarray) -> List[Detection]:
        """Detect using ONNX model (e.g., YOLO format)"""
        detections = []
        try:
            # Preprocess image
            input_size = (640, 640)
            img_resized = cv2.resize(image, input_size)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_normalized = img_rgb.astype(np.float32) / 255.0
            img_transposed = np.transpose(img_normalized, (2, 0, 1))
            img_batch = np.expand_dims(img_transposed, axis=0)
            
            # Run inference
            input_name = self.model.get_inputs()[0].name
            outputs = self.model.run(None, {input_name: img_batch})
            
            # Post-process (YOLO format)
            predictions = outputs[0][0]
            
            h, w = image.shape[:2]
            scale_x = w / input_size[0]
            scale_y = h / input_size[1]
            
            for pred in predictions:
                confidence = pred[4]
                if confidence >= self.confidence_threshold:
                    # Convert from center format to corner format
                    cx, cy, bw, bh = pred[0:4]
                    x1 = int((cx - bw/2) * scale_x)
                    y1 = int((cy - bh/2) * scale_y)
                    x2 = int((cx + bw/2) * scale_x)
                    y2 = int((cy + bh/2) * scale_y)
                    
                    # Clamp to image boundaries
                    x1 = max(0, min(x1, w))
                    y1 = max(0, min(y1, h))
                    x2 = max(0, min(x2, w))
                    y2 = max(0, min(y2, h))
                    
                    detections.append(Detection((x1, y1, x2, y2), float(confidence)))
        
        except Exception as e:
            print(f"ONNX detection error: {e}")
        
        return detections
    
    def _detect_hog(self, image: np.ndarray) -> List[Detection]:
        """Detect using HOG + SVM (fallback method)"""
        detections = []
        
        # Resize for faster processing on Orange Pi
        scale = min(1.0, 640 / max(image.shape[:2]))
        if scale < 1.0:
            resized = cv2.resize(image, None, fx=scale, fy=scale)
        else:
            resized = image
            scale = 1.0
        
        # Detect
        boxes, weights = self.hog.detectMultiScale(
            resized,
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.05
        )
        
        for (x, y, w, h), weight in zip(boxes, weights):
            # Scale back to original image size
            x1 = int(x / scale)
            y1 = int(y / scale)
            x2 = int((x + w) / scale)
            y2 = int((y + h) / scale)
            
            confidence = float(weight[0]) if isinstance(weight, np.ndarray) else float(weight)
            confidence = min(1.0, confidence)  # Normalize
            
            if confidence >= self.confidence_threshold:
                detections.append(Detection((x1, y1, x2, y2), confidence))
        
        return detections


class FaceDetector(BaseDetector):
    """Face detector using ONNX model or fallback Haar Cascade"""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        super().__init__(model_path, confidence_threshold)
        # Fallback Haar Cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.cascade = cv2.CascadeClassifier(cascade_path)
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """Detect faces in image"""
        if self.model is not None:
            return self._detect_onnx(image)
        else:
            return self._detect_cascade(image)
    
    def _detect_onnx(self, image: np.ndarray) -> List[Detection]:
        """Detect using ONNX model"""
        detections = []
        try:
            # Preprocess
            input_size = (320, 320)
            img_resized = cv2.resize(image, input_size)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_normalized = img_rgb.astype(np.float32) / 255.0
            img_transposed = np.transpose(img_normalized, (2, 0, 1))
            img_batch = np.expand_dims(img_transposed, axis=0)
            
            # Run inference
            input_name = self.model.get_inputs()[0].name
            outputs = self.model.run(None, {input_name: img_batch})
            
            # Post-process
            predictions = outputs[0][0]
            
            h, w = image.shape[:2]
            scale_x = w / input_size[0]
            scale_y = h / input_size[1]
            
            for pred in predictions:
                confidence = pred[4] if len(pred) > 4 else pred[2]
                if confidence >= self.confidence_threshold:
                    x1 = int(pred[0] * scale_x)
                    y1 = int(pred[1] * scale_y)
                    x2 = int(pred[2] * scale_x)
                    y2 = int(pred[3] * scale_y)
                    
                    # Clamp
                    x1 = max(0, min(x1, w))
                    y1 = max(0, min(y1, h))
                    x2 = max(0, min(x2, w))
                    y2 = max(0, min(y2, h))
                    
                    detections.append(Detection((x1, y1, x2, y2), float(confidence)))
        
        except Exception as e:
            print(f"ONNX face detection error: {e}")
        
        return detections
    
    def _detect_cascade(self, image: np.ndarray) -> List[Detection]:
        """Detect using Haar Cascade (fallback method)"""
        detections = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        for (x, y, w, h) in faces:
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            # Haar cascade doesn't provide confidence, use fixed value
            confidence = 0.9
            detections.append(Detection((x1, y1, x2, y2), confidence))
        
        return detections
