#!/usr/bin/env python3
"""
Chạy phân tích chi tiết cho video với thông số từng frame
"""

import argparse
import os
import sys
import time
import cv2

from src import config
from src.detector import PersonDetector, FaceDetector
from src.metrics import MetricsTracker


def analyze_video_detailed(video_path: str, output_dir: str, max_frames: int = None):
    """Phân tích video và hiển thị thông số chi tiết từng frame"""
    
    # Kiểm tra video tồn tại
    if not os.path.exists(video_path):
        print(f"Lỗi: Không tìm thấy video {video_path}")
        return
    
    # Tạo output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Khởi tạo detectors
    print("Đang khởi tạo models...")
    person_detector = PersonDetector(config.PERSON_MODEL_PATH, config.PERSON_CONFIDENCE_THRESHOLD)
    face_detector = FaceDetector(config.FACE_MODEL_PATH, config.FACE_CONFIDENCE_THRESHOLD)
    print("✓ Models đã sẵn sàng\n")
    
    # Mở video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video {video_path}")
        return
    
    # Lấy thông tin video
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("="*80)
    print(f"VIDEO: {os.path.basename(video_path)}")
    print(f"Tổng số frames: {total_frames}")
    print(f"FPS gốc: {video_fps:.2f}")
    print(f"Kích thước: {video_width}x{video_height}")
    print(f"Lọc person với accuracy > 0.87")
    print("="*80)
    print()
    
    # Tạo file log chi tiết
    log_path = os.path.join(output_dir, "detailed_analysis.txt")
    log_file = open(log_path, 'w', encoding='utf-8')
    log_file.write(f"CHI TIẾT PHÂN TÍCH VIDEO: {os.path.basename(video_path)}\n")
    log_file.write(f"Video FPS: {video_fps:.2f} | Kích thước: {video_width}x{video_height} | Tổng frames: {total_frames}\n")
    log_file.write(f"Lọc person với accuracy > 0.87\n")
    log_file.write("="*100 + "\n\n")
    
    # Metrics tracker
    metrics_tracker = MetricsTracker()
    metrics_tracker.start_processing()
    
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_start_time = time.time()
        
        # Lấy kích thước frame
        img_height, img_width = frame.shape[:2]
        img_size = f"{img_width}x{img_height}"
        
        # Phát hiện persons
        person_detect_start = time.time()
        person_detections = person_detector.detect(frame)
        person_detect_time = (time.time() - person_detect_start) * 1000  # ms
        
        # Lọc person với accuracy > 0.87
        filtered_persons = [p for p in person_detections if p.confidence > 0.87]
        
        # Tính accuracy trung bình của person detection (tất cả)
        avg_person_conf = sum(p.confidence for p in person_detections) / len(person_detections) if person_detections else 0.0
        
        # Tính accuracy trung bình của person đã lọc
        avg_filtered_conf = sum(p.confidence for p in filtered_persons) / len(filtered_persons) if filtered_persons else 0.0
        
        person_count = len(filtered_persons)
        face_count = 0
        total_face_detect_time = 0.0
        
        # Tính FPS tạm thời (sẽ cập nhật sau khi xử lý face)
        temp_frame_time = (time.time() - frame_start_time) * 1000  # ms
        temp_fps = 1000.0 / temp_frame_time if temp_frame_time > 0 else 0.0
        
        # Phát hiện faces trong từng person (chỉ với accuracy > 0.85)
        for person_idx, person_det in enumerate(filtered_persons):
            person_crop = person_det.get_crop(frame)
            if person_crop.size == 0:
                continue
            
            # Lưu person crop
            person_height, person_width = person_crop.shape[:2]
            person_size = f"{person_width}x{person_height}"
            person_img_path = os.path.join(output_dir, f"frame_{frame_idx:04d}_person_{person_idx}.jpg")
            cv2.imwrite(person_img_path, person_crop)
            
            # Lưu thông số person vào file txt
            person_txt_path = os.path.join(output_dir, f"frame_{frame_idx:04d}_person_{person_idx}.txt")
            with open(person_txt_path, 'w', encoding='utf-8') as pf:
                pf.write(f"Frame: {frame_idx}\n")
                pf.write(f"Số đối tượng: 1 person\n")
                pf.write(f"Thời gian xử lý (person detection): {person_detect_time:.2f} ms\n")
                pf.write(f"Inference time (person): {person_detect_time:.2f} ms\n")
                pf.write(f"Accuracy (person): {person_det.confidence:.4f}\n")
                pf.write(f"Kích thước ảnh crop: {person_size}\n")
                pf.write(f"Kích thước frame gốc: {img_size}\n")
                pf.write(f"FPS (tại thời điểm này): {temp_fps:.2f}\n")
            
            # Phát hiện faces trong person crop
            face_detect_start = time.time()
            face_detections = face_detector.detect(person_crop)
            face_detect_time = (time.time() - face_detect_start) * 1000  # ms
            total_face_detect_time += face_detect_time
            
            # Lưu face crops
            for face_idx, face_det in enumerate(face_detections):
                face_crop = face_det.get_crop(person_crop)
                if face_crop.size == 0:
                    continue
                
                # Lưu face crop
                face_height, face_width = face_crop.shape[:2]
                face_size = f"{face_width}x{face_height}"
                face_img_path = os.path.join(output_dir, f"frame_{frame_idx:04d}_person_{person_idx}_face_{face_idx}.jpg")
                cv2.imwrite(face_img_path, face_crop)
                
                # Tính FPS hiện tại
                current_frame_time = (time.time() - frame_start_time) * 1000
                current_fps = 1000.0 / current_frame_time if current_frame_time > 0 else 0.0
                
                # Lưu thông số face vào file txt
                face_txt_path = os.path.join(output_dir, f"frame_{frame_idx:04d}_person_{person_idx}_face_{face_idx}.txt")
                with open(face_txt_path, 'w', encoding='utf-8') as ff:
                    ff.write(f"Frame: {frame_idx}\n")
                    ff.write(f"Số đối tượng: 1 face\n")
                    ff.write(f"Thời gian xử lý (face detection): {face_detect_time:.2f} ms\n")
                    ff.write(f"Inference time (face): {face_detect_time:.2f} ms\n")
                    ff.write(f"Accuracy (face): {face_det.confidence:.4f}\n")
                    ff.write(f"Kích thước ảnh crop: {face_size}\n")
                    ff.write(f"Kích thước person crop: {person_size}\n")
                    ff.write(f"Kích thước frame gốc: {img_size}\n")
                    ff.write(f"FPS (tại thời điểm này): {current_fps:.2f}\n")
            
            face_count += len(face_detections)
        
        # Tính tổng thời gian xử lý frame
        frame_time = (time.time() - frame_start_time) * 1000  # ms
        
        # Tính inference time (chỉ detection, không gồm crop/save)
        inference_time = person_detect_time + total_face_detect_time
        
        # Tính FPS thực tế
        current_fps = 1000.0 / frame_time if frame_time > 0 else 0.0
        
        # Hiển thị thông tin frame
        print(f"Frame {frame_idx:04d}:")
        print(f"  - Tổng persons phát hiện: {len(person_detections)} (accuracy > 0.87: {person_count})")
        print(f"  - Số đối tượng được lưu: {person_count} persons, {face_count} faces")
        print(f"  - Thời gian xử lý: {frame_time:.2f} ms")
        print(f"  - Inference time: {inference_time:.2f} ms")
        print(f"  - Accuracy trung bình (all): {avg_person_conf:.4f}")
        print(f"  - Accuracy trung bình (filtered): {avg_filtered_conf:.4f}")
        print(f"  - Kích thước ảnh: {img_size}")
        print(f"  - FPS: {current_fps:.2f}")
        print()
        
        # Ghi vào log file
        log_file.write(f"Frame {frame_idx:04d}:\n")
        log_file.write(f"  Tổng persons phát hiện: {len(person_detections)} (accuracy > 0.87: {person_count})\n")
        log_file.write(f"  Số đối tượng được lưu: {person_count} persons, {face_count} faces\n")
        log_file.write(f"  Thời gian xử lý: {frame_time:.2f} ms\n")
        log_file.write(f"  Inference time: {inference_time:.2f} ms\n")
        log_file.write(f"  Accuracy trung bình (all): {avg_person_conf:.4f}\n")
        log_file.write(f"  Accuracy trung bình (filtered): {avg_filtered_conf:.4f}\n")
        log_file.write(f"  Kích thước ảnh: {img_size}\n")
        log_file.write(f"  FPS: {current_fps:.2f}\n")
        log_file.write("\n")
        
        # Track metrics
        metrics_tracker.add_frame_metrics(current_fps, avg_person_conf, frame_time / 1000.0)
        
        frame_idx += 1
        
        # Kiểm tra max_frames nếu được set
        if max_frames is not None and frame_idx >= max_frames:
            print(f"\n⚠️  Đã xử lý đủ {max_frames} frames, dừng lại...")
            break
    
    cap.release()
    metrics_tracker.end_processing()
    
    # Tính toán và hiển thị tổng kết
    print("="*80)
    print("TỔNG KẾT")
    print("="*80)
    
    summary = metrics_tracker.get_summary()
    
    print(f"Tổng số frames: {frame_idx}")
    print(f"Tổng thời gian xử lý: {summary['total_processing_time']:.2f}s")
    print(f"FPS trung bình: {summary['fps_avg']:.2f}")
    print(f"FPS cao nhất: {summary['fps_max']:.2f}")
    print(f"FPS thấp nhất: {summary['fps_min']:.2f}")
    print(f"Accuracy trung bình: {summary['accuracy_avg']:.4f}")
    print(f"Thời gian xử lý trung bình/frame: {summary['person_processing_time_avg']*1000:.2f} ms")
    
    # Ghi tổng kết vào log
    log_file.write("="*100 + "\n")
    log_file.write("TỔNG KẾT\n")
    log_file.write("="*100 + "\n")
    log_file.write(f"Tổng số frames: {frame_idx}\n")
    log_file.write(f"Tổng thời gian xử lý: {summary['total_processing_time']:.2f}s\n")
    log_file.write(f"FPS trung bình: {summary['fps_avg']:.2f}\n")
    log_file.write(f"FPS cao nhất: {summary['fps_max']:.2f}\n")
    log_file.write(f"FPS thấp nhất: {summary['fps_min']:.2f}\n")
    log_file.write(f"Accuracy trung bình: {summary['accuracy_avg']:.4f}\n")
    log_file.write(f"Thời gian xử lý trung bình/frame: {summary['person_processing_time_avg']*1000:.2f} ms\n")
    
    log_file.close()
    
    print(f"\n✓ Chi tiết đã được lưu vào: {log_path}")


def main():
    parser = argparse.ArgumentParser(description='Phân tích chi tiết video')
    parser.add_argument('--input', '-i', type=str, required=True, help='Đường dẫn video')
    parser.add_argument('--output', '-o', type=str, default='output/analysis', help='Thư mục output')
    parser.add_argument('--max-frames', '-m', type=int, default=None, help='Số frame tối đa để test (None = full video)')
    
    args = parser.parse_args()
    
    analyze_video_detailed(args.input, args.output, args.max_frames)


if __name__ == '__main__':
    main()
