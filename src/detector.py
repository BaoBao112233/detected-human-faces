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
        """Detect using ONNX model (supports NanoDet, YOLO, etc.)"""
        detections = []
        try:
            # Get input shape from model
            input_shape = self.model.get_inputs()[0].shape
            # Handle dynamic dimensions (e.g., ['batch', 3, 416, 416] or [1, 3, 416, 416])
            if len(input_shape) >= 4:
                input_height = input_shape[2] if isinstance(input_shape[2], int) else 416
                input_width = input_shape[3] if isinstance(input_shape[3], int) else 416
            else:
                input_height, input_width = 416, 416
            
            input_size = (input_width, input_height)
            
            # Preprocess image
            img_resized = cv2.resize(image, input_size)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_normalized = img_rgb.astype(np.float32) / 255.0
            img_transposed = np.transpose(img_normalized, (2, 0, 1))
            img_batch = np.expand_dims(img_transposed, axis=0)
            
            # Run inference
            input_name = self.model.get_inputs()[0].name
            outputs = self.model.run(None, {input_name: img_batch})
            
            h, w = image.shape[:2]
            scale_x = w / input_size[0]
            scale_y = h / input_size[1]
            
            # Detect model type by output structure
            if len(outputs) == 6 and outputs[0].shape[-1] == 80:
                # NanoDet format: 6 outputs (3 cls + 3 bbox)
                # outputs[0,1,2]: class scores for 3 scales
                # outputs[3,4,5]: bbox predictions for 3 scales
                detections = self._parse_nanodet_output(outputs, scale_x, scale_y, w, h)
            else:
                # Try YOLO-style format
                detections = self._parse_yolo_output(outputs, scale_x, scale_y, w, h)
                
        except Exception as e:
            print(f"ONNX detection error: {e}")
            
        return detections
    
    def _parse_nanodet_output(self, outputs, scale_x, scale_y, img_w, img_h):
        """Parse NanoDet model output (COCO 80 classes)"""
        detections = []
        
        # NanoDet outputs: [cls_0, cls_1, cls_2, bbox_0, bbox_1, bbox_2]
        # Person class in COCO is index 0
        person_class_id = 0
        
        # Process each scale
        for scale_idx in range(3):
            cls_pred = outputs[scale_idx][0]  # Shape: [num_anchors, 80]
            bbox_pred = outputs[scale_idx + 3][0]  # Shape: [num_anchors, 32]
            
            # Get stride for this scale
            num_anchors = cls_pred.shape[0]
            if num_anchors == 2704:  # 52x52 = 2704
                stride = 8
                grid_h, grid_w = 52, 52
            elif num_anchors == 676:  # 26x26 = 676  
                stride = 16
                grid_h, grid_w = 26, 26
            elif num_anchors == 169:  # 13x13 = 169
                stride = 32
                grid_h, grid_w = 13, 13
            else:
                continue
            
            # Process each anchor
            for idx in range(num_anchors):
                # Get person class score
                person_score = cls_pred[idx, person_class_id]
                
                if person_score >= self.confidence_threshold:
                    # Calculate grid position
                    grid_y = idx // grid_w
                    grid_x = idx % grid_w
                    
                    # Decode bbox using DFL (Distribution Focal Loss)
                    # bbox_pred shape: [32] = [8 bins for each of 4 distances: left, top, right, bottom]
                    reg_max = 7  # NanoDet uses reg_max=7 (8 bins: 0-7)
                    
                    # Softmax over bins for each distance
                    bbox_dist = bbox_pred[idx].reshape(4, 8)  # [4 directions, 8 bins]
                    
                    # Apply softmax to get distribution
                    bbox_dist_exp = np.exp(bbox_dist - np.max(bbox_dist, axis=1, keepdims=True))
                    bbox_dist_softmax = bbox_dist_exp / np.sum(bbox_dist_exp, axis=1, keepdims=True)
                    
                    # Calculate expected value (weighted sum)
                    bin_range = np.arange(8).astype(np.float32)
                    distances = np.sum(bbox_dist_softmax * bin_range, axis=1)  # [left, top, right, bottom]
                    
                    # Anchor center in original 416x416 space
                    cx = (grid_x + 0.5) * stride
                    cy = (grid_y + 0.5) * stride
                    
                    # Decode bbox (distances are in stride units)
                    x1 = cx - distances[0] * stride
                    y1 = cy - distances[1] * stride
                    x2 = cx + distances[2] * stride
                    y2 = cy + distances[3] * stride
                    
                    # Scale to original image size
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)
                    
                    # Clamp to image boundaries
                    x1 = max(0, min(x1, img_w))
                    y1 = max(0, min(y1, img_h))
                    x2 = max(0, min(x2, img_w))
                    y2 = max(0, min(y2, img_h))
                    
                    if x2 > x1 and y2 > y1:
                        detections.append(Detection((x1, y1, x2, y2), float(person_score)))
        
        return detections
    
    def _parse_yolo_output(self, outputs, scale_x, scale_y, img_w, img_h):
        """Parse YOLO-style model output"""
        detections = []
        
        # Post-process (YOLO format)
        predictions = outputs[0][0]
        
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
                x1 = max(0, min(x1, img_w))
                y1 = max(0, min(y1, img_h))
                x2 = max(0, min(x2, img_w))
                y2 = max(0, min(y2, img_h))
                
                if x2 > x1 and y2 > y1:
                    detections.append(Detection((x1, y1, x2, y2), float(confidence)))
        
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
            # Get input shape from model
            input_shape = self.model.get_inputs()[0].shape
            # Handle dynamic dimensions
            if len(input_shape) >= 4:
                input_height = input_shape[2] if isinstance(input_shape[2], int) else 320
                input_width = input_shape[3] if isinstance(input_shape[3], int) else 320
            else:
                input_height, input_width = 320, 320
            
            input_size = (input_width, input_height)
            
            # Preprocess
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
