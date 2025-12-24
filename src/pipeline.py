"""
Pipeline implementations for person and face detection
Supports both parallel and sequential processing
"""

import cv2
import numpy as np
import time
import os
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .detector import PersonDetector, FaceDetector, Detection
from .metrics import MetricsTracker, FrameMetrics
from . import config


class ProcessingPipeline:
    """Base class for processing pipelines"""
    
    def __init__(self, person_detector: PersonDetector, face_detector: FaceDetector):
        self.person_detector = person_detector
        self.face_detector = face_detector
        self.metrics_tracker = MetricsTracker()
    
    def process_image(self, image: np.ndarray, output_prefix: str) -> Tuple[int, int]:
        """Process a single image - to be implemented by subclasses"""
        raise NotImplementedError
    
    def process_video(self, video_path: str, output_prefix: str):
        """Process video file"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path}")
        print(f"Total frames: {total_frames}, FPS: {fps}")
        
        self.metrics_tracker.start_processing()
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_output_prefix = f"{output_prefix}_frame_{frame_idx:06d}"
            self.process_image(frame, frame_output_prefix)
            
            frame_idx += 1
            if frame_idx % 10 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames")
        
        cap.release()
        self.metrics_tracker.end_processing()
        
        print(f"\nVideo processing complete!")
        self.metrics_tracker.print_summary()
        
        # Save summary
        summary_path = os.path.join(config.LOG_DIR, f"{os.path.basename(output_prefix)}_summary.txt")
        self.metrics_tracker.save_to_file(summary_path)
        print(f"Summary saved to: {summary_path}")


class SequentialPipeline(ProcessingPipeline):
    """Pipeline that processes person detection first, then face detection"""
    
    def process_image(self, image: np.ndarray, output_prefix: str) -> Tuple[int, int]:
        """Process image: detect persons first, then detect faces in each person crop"""
        frame_start_time = time.time()
        
        # Get image size
        img_height, img_width = image.shape[:2]
        img_size = f"{img_width}x{img_height}"
        
        # Step 1: Detect persons
        person_detect_start = time.time()
        person_detections = self.person_detector.detect(image)
        person_detect_time = (time.time() - person_detect_start) * 1000  # Convert to ms
        
        # Save annotated frame with bounding boxes
        annotated_frame = image.copy()
        for person_det in person_detections:
            x1, y1, x2, y2 = person_det.bbox
            # Draw bbox (red for person)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # Draw confidence
            conf_text = f"{person_det.confidence:.2f}"
            cv2.putText(annotated_frame, conf_text, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Save annotated frame
        annotated_path = f"{output_prefix}_annotated.jpg"
        cv2.imwrite(annotated_path, annotated_frame)
        
        # Calculate average confidence for all persons
        avg_person_conf = sum(p.confidence for p in person_detections) / len(person_detections) if person_detections else 0.0
        
        person_count = 0
        face_count = 0
        
        # Step 2: For each person, detect faces
        for person_idx, person_det in enumerate(person_detections):
            # Crop person
            person_crop = person_det.get_crop(image)
            if person_crop.size == 0:
                continue
            
            # Get person crop size
            person_height, person_width = person_crop.shape[:2]
            person_size = f"{person_width}x{person_height}"
            
            # Save person crop
            person_path = f"{output_prefix}_person_{person_idx}.jpg"
            cv2.imwrite(person_path, person_crop)
            
            # Detect faces in person crop
            face_detect_start = time.time()
            face_detections = self.face_detector.detect(person_crop)
            face_detect_time = (time.time() - face_detect_start) * 1000  # Convert to ms
            
            # Save face crops
            for face_idx, face_det in enumerate(face_detections):
                face_crop = face_det.get_crop(person_crop)
                if face_crop.size == 0:
                    continue
                
                face_height, face_width = face_crop.shape[:2]
                face_size = f"{face_width}x{face_height}"
                
                face_path = f"{output_prefix}_person_{person_idx}_face_{face_idx}.jpg"
                cv2.imwrite(face_path, face_crop)
                
                face_count += 1
            
            person_count += 1
        
        # Calculate total frame time and FPS
        frame_time = (time.time() - frame_start_time) * 1000  # Convert to ms
        current_fps = 1000.0 / frame_time if frame_time > 0 else 0.0
        
        # Log detailed metrics for this frame
        print(f"[Frame] Objects: {person_count} persons, {face_count} faces | "
              f"Time: {frame_time:.1f}ms | Accuracy: {avg_person_conf:.3f} | "
              f"Size: {img_size} | FPS: {current_fps:.2f}")
        
        # Track metrics
        self.metrics_tracker.add_frame_metrics(current_fps, avg_person_conf, frame_time / 1000.0)
        
        return person_count, face_count


class ParallelPipeline(ProcessingPipeline):
    """Pipeline that processes person and face detection in parallel"""
    
    def __init__(self, person_detector: PersonDetector, face_detector: FaceDetector, max_workers: int = 2):
        super().__init__(person_detector, face_detector)
        self.max_workers = max_workers
    
    def process_image(self, image: np.ndarray, output_prefix: str) -> Tuple[int, int]:
        """Process image: detect persons and faces in parallel"""
        frame_start_time = time.time()
        
        person_detections = []
        face_detections = []
        
        # Run person and face detection in parallel on the full image
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            person_future = executor.submit(self.person_detector.detect, image)
            face_future = executor.submit(self.face_detector.detect, image)
            
            person_detections = person_future.result()
            face_detections = face_future.result()
        
        person_count = 0
        face_count = 0
        
        # Save person crops
        for person_idx, person_det in enumerate(person_detections):
            person_start_time = time.time()
            
            person_crop = person_det.get_crop(image)
            if person_crop.size == 0:
                continue
            
            person_path = f"{output_prefix}_person_{person_idx}.jpg"
            cv2.imwrite(person_path, person_crop)
            
            person_time = time.time() - person_start_time
            frame_time = time.time() - frame_start_time
            current_fps = 1.0 / frame_time if frame_time > 0 else 0.0
            
            # Save person metrics
            metrics = FrameMetrics(current_fps, person_det.confidence)
            metrics_path = person_path.replace('.jpg', '.txt')
            metrics.save_to_file(metrics_path)
            
            person_count += 1
        
        # Save face crops
        for face_idx, face_det in enumerate(face_detections):
            face_crop = face_det.get_crop(image)
            if face_crop.size == 0:
                continue
            
            face_path = f"{output_prefix}_face_{face_idx}.jpg"
            cv2.imwrite(face_path, face_crop)
            
            frame_time = time.time() - frame_start_time
            current_fps = 1.0 / frame_time if frame_time > 0 else 0.0
            
            # Save face metrics
            metrics = FrameMetrics(current_fps, face_det.confidence)
            metrics_path = face_path.replace('.jpg', '.txt')
            metrics.save_to_file(metrics_path)
            
            # Track metrics
            person_time = frame_time / max(len(person_detections), 1)
            self.metrics_tracker.add_frame_metrics(current_fps, face_det.confidence, person_time)
            
            face_count += 1
        
        return person_count, face_count


def create_pipeline(pipeline_mode: str, person_model_path: str, face_model_path: str, 
                   person_threshold: float = 0.5, face_threshold: float = 0.5) -> ProcessingPipeline:
    """Factory function to create a pipeline"""
    
    # Initialize detectors
    person_detector = PersonDetector(person_model_path, person_threshold)
    face_detector = FaceDetector(face_model_path, face_threshold)
    
    # Create pipeline based on mode
    if pipeline_mode == config.PIPELINE_PARALLEL:
        print("Using PARALLEL pipeline (person and face detection in parallel)")
        return ParallelPipeline(person_detector, face_detector, max_workers=config.MAX_THREADS)
    elif pipeline_mode == config.PIPELINE_SEQUENTIAL:
        print("Using SEQUENTIAL pipeline (person first, then face)")
        return SequentialPipeline(person_detector, face_detector)
    else:
        raise ValueError(f"Unknown pipeline mode: {pipeline_mode}")
