#!/usr/bin/env python3
"""
Main entry point for human and face detection system
Optimized for Orange Pi RV 2 (4GB RAM)

Usage:
    python main.py --input input/image.jpg --pipeline sequential
    python main.py --input input/video.mp4 --pipeline parallel --person-model models/yolov8n.onnx
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2

import config
from pipeline import create_pipeline


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Human and Face Detection System for Orange Pi',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process image with sequential pipeline
  python main.py --input input/photo.jpg --pipeline sequential
  
  # Process video with parallel pipeline
  python main.py --input input/video.mp4 --pipeline parallel
  
  # Use custom models
  python main.py --input input/test.jpg --person-model models/custom_person.onnx --face-model models/custom_face.onnx
  
  # Adjust confidence thresholds
  python main.py --input input/test.jpg --person-threshold 0.6 --face-threshold 0.7
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input image or video file'
    )
    
    parser.add_argument(
        '--pipeline', '-p',
        type=str,
        choices=['sequential', 'parallel'],
        default=config.DEFAULT_PIPELINE,
        help=f'Pipeline mode: sequential (person->face) or parallel (person||face). Default: {config.DEFAULT_PIPELINE}'
    )
    
    parser.add_argument(
        '--person-model',
        type=str,
        default=config.PERSON_MODEL_PATH,
        help=f'Path to person detection model (ONNX format). Default: {config.PERSON_MODEL_PATH}'
    )
    
    parser.add_argument(
        '--face-model',
        type=str,
        default=config.FACE_MODEL_PATH,
        help=f'Path to face detection model (ONNX format). Default: {config.FACE_MODEL_PATH}'
    )
    
    parser.add_argument(
        '--person-threshold',
        type=float,
        default=config.PERSON_CONFIDENCE_THRESHOLD,
        help=f'Confidence threshold for person detection (0-1). Default: {config.PERSON_CONFIDENCE_THRESHOLD}'
    )
    
    parser.add_argument(
        '--face-threshold',
        type=float,
        default=config.FACE_CONFIDENCE_THRESHOLD,
        help=f'Confidence threshold for face detection (0-1). Default: {config.FACE_CONFIDENCE_THRESHOLD}'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=config.OUTPUT_DIR,
        help=f'Directory to save output files. Default: {config.OUTPUT_DIR}'
    )
    
    return parser.parse_args()


def validate_input_file(input_path: str) -> str:
    """Validate input file exists and determine type"""
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    ext = os.path.splitext(input_path)[1].lower()
    
    if ext in config.SUPPORTED_IMAGE_FORMATS:
        return 'image'
    elif ext in config.SUPPORTED_VIDEO_FORMATS:
        return 'video'
    else:
        print(f"Error: Unsupported file format: {ext}")
        print(f"Supported image formats: {config.SUPPORTED_IMAGE_FORMATS}")
        print(f"Supported video formats: {config.SUPPORTED_VIDEO_FORMATS}")
        sys.exit(1)


def main():
    """Main function"""
    args = parse_arguments()
    
    print("="*60)
    print("Human and Face Detection System")
    print("Optimized for Orange Pi RV 2")
    print("="*60)
    
    # Validate input
    input_type = validate_input_file(args.input)
    print(f"\nInput file: {args.input}")
    print(f"Input type: {input_type}")
    print(f"Pipeline mode: {args.pipeline}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create output prefix
    input_basename = os.path.splitext(os.path.basename(args.input))[0]
    output_prefix = os.path.join(args.output_dir, input_basename)
    
    # Create pipeline
    print(f"\nInitializing pipeline...")
    print(f"Person model: {args.person_model}")
    print(f"Face model: {args.face_model}")
    print(f"Person threshold: {args.person_threshold}")
    print(f"Face threshold: {args.face_threshold}")
    
    try:
        pipeline = create_pipeline(
            pipeline_mode=args.pipeline,
            person_model_path=args.person_model,
            face_model_path=args.face_model,
            person_threshold=args.person_threshold,
            face_threshold=args.face_threshold
        )
    except Exception as e:
        print(f"\nError initializing pipeline: {e}")
        sys.exit(1)
    
    # Process input
    print(f"\nStarting processing...")
    start_time = time.time()
    
    try:
        if input_type == 'image':
            # Process single image
            image = cv2.imread(args.input)
            if image is None:
                print(f"Error: Failed to load image: {args.input}")
                sys.exit(1)
            
            # Resize if too large (for Orange Pi optimization)
            h, w = image.shape[:2]
            if w > config.MAX_INPUT_WIDTH or h > config.MAX_INPUT_HEIGHT:
                scale = min(config.MAX_INPUT_WIDTH / w, config.MAX_INPUT_HEIGHT / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                image = cv2.resize(image, (new_w, new_h))
                print(f"Resized image from {w}x{h} to {new_w}x{new_h}")
            
            pipeline.metrics_tracker.start_processing()
            person_count, face_count = pipeline.process_image(image, output_prefix)
            pipeline.metrics_tracker.end_processing()
            
            print(f"\nImage processing complete!")
            print(f"Detected {person_count} person(s) and {face_count} face(s)")
            pipeline.metrics_tracker.print_summary()
            
            # Save summary
            summary_path = os.path.join(config.LOG_DIR, f"{input_basename}_summary.txt")
            pipeline.metrics_tracker.save_to_file(summary_path)
            print(f"Summary saved to: {summary_path}")
            
        elif input_type == 'video':
            # Process video
            pipeline.process_video(args.input, output_prefix)
    
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f}s")
    print(f"Output files saved to: {args.output_dir}")
    print("\nDone!")


if __name__ == "__main__":
    main()
