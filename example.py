#!/usr/bin/env python3
"""
Example script showing how to use the detection system programmatically
"""

import cv2
import os
from src.pipeline import create_pipeline
from src import config

def example_process_image():
    """Example: Process a single image"""
    print("Example 1: Process single image")
    
    # Create pipeline
    pipeline = create_pipeline(
        pipeline_mode=config.PIPELINE_SEQUENTIAL,
        person_model_path=config.PERSON_MODEL_PATH,
        face_model_path=config.FACE_MODEL_PATH
    )
    
    # Load image
    image_path = "input/test.jpg"
    if not os.path.exists(image_path):
        print(f"Please place a test image at {image_path}")
        return
    
    image = cv2.imread(image_path)
    
    # Process
    output_prefix = "output/example_test"
    pipeline.metrics_tracker.start_processing()
    person_count, face_count = pipeline.process_image(image, output_prefix)
    pipeline.metrics_tracker.end_processing()
    
    print(f"Detected {person_count} persons and {face_count} faces")
    pipeline.metrics_tracker.print_summary()


def example_compare_pipelines():
    """Example: Compare sequential vs parallel pipeline"""
    print("\nExample 2: Compare pipeline performance")
    
    image_path = "input/test.jpg"
    if not os.path.exists(image_path):
        print(f"Please place a test image at {image_path}")
        return
    
    image = cv2.imread(image_path)
    
    # Test sequential
    print("\n--- Testing Sequential Pipeline ---")
    pipeline_seq = create_pipeline(
        pipeline_mode=config.PIPELINE_SEQUENTIAL,
        person_model_path=config.PERSON_MODEL_PATH,
        face_model_path=config.FACE_MODEL_PATH
    )
    pipeline_seq.metrics_tracker.start_processing()
    pipeline_seq.process_image(image, "output/seq_test")
    pipeline_seq.metrics_tracker.end_processing()
    seq_summary = pipeline_seq.metrics_tracker.get_summary()
    
    # Test parallel
    print("\n--- Testing Parallel Pipeline ---")
    pipeline_par = create_pipeline(
        pipeline_mode=config.PIPELINE_PARALLEL,
        person_model_path=config.PERSON_MODEL_PATH,
        face_model_path=config.FACE_MODEL_PATH
    )
    pipeline_par.metrics_tracker.start_processing()
    pipeline_par.process_image(image, "output/par_test")
    pipeline_par.metrics_tracker.end_processing()
    par_summary = pipeline_par.metrics_tracker.get_summary()
    
    # Compare
    print("\n--- Comparison ---")
    print(f"Sequential - Time: {seq_summary['total_processing_time']:.2f}s, Avg FPS: {seq_summary['fps_avg']:.2f}")
    print(f"Parallel   - Time: {par_summary['total_processing_time']:.2f}s, Avg FPS: {par_summary['fps_avg']:.2f}")


def example_custom_thresholds():
    """Example: Use custom confidence thresholds"""
    print("\nExample 3: Custom confidence thresholds")
    
    # Create pipeline with higher thresholds (fewer but more confident detections)
    pipeline = create_pipeline(
        pipeline_mode=config.PIPELINE_SEQUENTIAL,
        person_model_path=config.PERSON_MODEL_PATH,
        face_model_path=config.FACE_MODEL_PATH,
        person_threshold=0.7,  # Higher threshold
        face_threshold=0.8     # Higher threshold
    )
    
    image_path = "input/test.jpg"
    if not os.path.exists(image_path):
        print(f"Please place a test image at {image_path}")
        return
    
    image = cv2.imread(image_path)
    
    pipeline.metrics_tracker.start_processing()
    person_count, face_count = pipeline.process_image(image, "output/high_threshold_test")
    pipeline.metrics_tracker.end_processing()
    
    print(f"With high thresholds: {person_count} persons, {face_count} faces")


if __name__ == "__main__":
    # Run examples
    example_process_image()
    # example_compare_pipelines()
    # example_custom_thresholds()
