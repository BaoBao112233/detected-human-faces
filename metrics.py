"""
Metrics tracking for detection performance
Tracks FPS, accuracy, and processing time
"""

import time
from typing import List, Dict
import numpy as np


class MetricsTracker:
    """Track performance metrics during detection"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.fps_values = []
        self.accuracy_values = []
        self.person_processing_times = []
        self.total_processing_time = 0.0
        self.frame_count = 0
        self.start_time = None
    
    def start_processing(self):
        """Mark the start of processing"""
        self.start_time = time.time()
    
    def end_processing(self):
        """Mark the end of processing"""
        if self.start_time is not None:
            self.total_processing_time = time.time() - self.start_time
    
    def add_frame_metrics(self, fps: float, accuracy: float, person_time: float = 0.0):
        """Add metrics for a single frame/person"""
        self.fps_values.append(fps)
        self.accuracy_values.append(accuracy)
        if person_time > 0:
            self.person_processing_times.append(person_time)
        self.frame_count += 1
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        summary = {
            'total_frames': self.frame_count,
            'total_processing_time': self.total_processing_time,
            'fps_max': max(self.fps_values) if self.fps_values else 0.0,
            'fps_min': min(self.fps_values) if self.fps_values else 0.0,
            'fps_avg': np.mean(self.fps_values) if self.fps_values else 0.0,
            'accuracy_max': max(self.accuracy_values) if self.accuracy_values else 0.0,
            'accuracy_min': min(self.accuracy_values) if self.accuracy_values else 0.0,
            'accuracy_avg': np.mean(self.accuracy_values) if self.accuracy_values else 0.0,
            'person_processing_time_avg': np.mean(self.person_processing_times) if self.person_processing_times else 0.0,
        }
        return summary
    
    def print_summary(self):
        """Print summary to console"""
        summary = self.get_summary()
        print("\n" + "="*60)
        print("PERFORMANCE REPORT")
        print("="*60)
        print(f"Total frames processed: {summary['total_frames']}")
        print(f"Total processing time: {summary['total_processing_time']:.2f}s")
        print(f"Average person processing time: {summary['person_processing_time_avg']*1000:.2f}ms")
        print(f"\nFPS Statistics:")
        print(f"  Max FPS: {summary['fps_max']:.2f}")
        print(f"  Min FPS: {summary['fps_min']:.2f}")
        print(f"  Avg FPS: {summary['fps_avg']:.2f}")
        print(f"\nAccuracy Statistics:")
        print(f"  Max Accuracy: {summary['accuracy_max']:.4f}")
        print(f"  Min Accuracy: {summary['accuracy_min']:.4f}")
        print(f"  Avg Accuracy: {summary['accuracy_avg']:.4f}")
        print("="*60 + "\n")
    
    def save_to_file(self, filepath: str):
        """Save summary to text file"""
        summary = self.get_summary()
        with open(filepath, 'w') as f:
            f.write("PERFORMANCE REPORT\n")
            f.write("="*60 + "\n")
            f.write(f"Total frames processed: {summary['total_frames']}\n")
            f.write(f"Total processing time: {summary['total_processing_time']:.2f}s\n")
            f.write(f"Average person processing time: {summary['person_processing_time_avg']*1000:.2f}ms\n")
            f.write(f"\nFPS Statistics:\n")
            f.write(f"  Max FPS: {summary['fps_max']:.2f}\n")
            f.write(f"  Min FPS: {summary['fps_min']:.2f}\n")
            f.write(f"  Avg FPS: {summary['fps_avg']:.2f}\n")
            f.write(f"\nAccuracy Statistics:\n")
            f.write(f"  Max Accuracy: {summary['accuracy_max']:.4f}\n")
            f.write(f"  Min Accuracy: {summary['accuracy_min']:.4f}\n")
            f.write(f"  Avg Accuracy: {summary['accuracy_avg']:.4f}\n")


class FrameMetrics:
    """Metrics for a single frame/detection"""
    
    def __init__(self, fps: float, accuracy: float):
        self.fps = fps
        self.accuracy = accuracy
    
    def to_text(self) -> str:
        """Convert metrics to text format"""
        return f"fps: {self.fps:.2f}\nacc: {self.accuracy:.4f}\n"
    
    def save_to_file(self, filepath: str):
        """Save metrics to text file"""
        with open(filepath, 'w') as f:
            f.write(self.to_text())
