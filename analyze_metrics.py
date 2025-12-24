#!/usr/bin/env python3
"""
Analyze metrics from detection log file
Shows detailed statistics for each frame
"""

import sys
import re
from collections import defaultdict

def parse_log(log_file):
    """Parse log file and extract frame metrics"""
    frames = []
    pattern = r'\[Frame\] Objects: (\d+) persons, (\d+) faces \| Time: ([\d.]+)ms \| Accuracy: ([\d.]+) \| Size: (\d+x\d+) \| FPS: ([\d.]+)'
    
    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                persons, faces, time_ms, accuracy, size, fps = match.groups()
                frames.append({
                    'persons': int(persons),
                    'faces': int(faces),
                    'time_ms': float(time_ms),
                    'accuracy': float(accuracy),
                    'size': size,
                    'fps': float(fps)
                })
    
    return frames

def analyze_frames(frames):
    """Analyze and print statistics"""
    if not frames:
        print("No frames found in log")
        return
    
    print("=" * 80)
    print(f"DETAILED METRICS ANALYSIS - Total Frames: {len(frames)}")
    print("=" * 80)
    print()
    
    # Overall stats
    total_persons = sum(f['persons'] for f in frames)
    total_faces = sum(f['faces'] for f in frames)
    avg_time = sum(f['time_ms'] for f in frames) / len(frames)
    avg_fps = sum(f['fps'] for f in frames) / len(frames)
    
    # Frames with detections
    frames_with_persons = [f for f in frames if f['persons'] > 0]
    avg_accuracy = sum(f['accuracy'] for f in frames_with_persons) / len(frames_with_persons) if frames_with_persons else 0
    
    print(f"📊 OVERALL STATISTICS")
    print(f"  Total Objects Detected: {total_persons} persons, {total_faces} faces")
    print(f"  Frames with Detections: {len(frames_with_persons)}/{len(frames)} ({len(frames_with_persons)*100/len(frames):.1f}%)")
    print(f"  Average Processing Time: {avg_time:.1f}ms")
    print(f"  Average FPS: {avg_fps:.2f}")
    print(f"  Average Accuracy (when detected): {avg_accuracy:.3f}")
    print(f"  Frame Size: {frames[0]['size']}")
    print()
    
    # Processing time ranges
    print(f"⏱️  PROCESSING TIME BREAKDOWN")
    time_ranges = {
        '< 200ms (>5 FPS)': [f for f in frames if f['time_ms'] < 200],
        '200-500ms (2-5 FPS)': [f for f in frames if 200 <= f['time_ms'] < 500],
        '500-1000ms (1-2 FPS)': [f for f in frames if 500 <= f['time_ms'] < 1000],
        '1000-2000ms (0.5-1 FPS)': [f for f in frames if 1000 <= f['time_ms'] < 2000],
        '> 2000ms (<0.5 FPS)': [f for f in frames if f['time_ms'] >= 2000]
    }
    
    for range_name, range_frames in time_ranges.items():
        count = len(range_frames)
        pct = count * 100 / len(frames)
        avg_persons = sum(f['persons'] for f in range_frames) / count if count > 0 else 0
        print(f"  {range_name:25s}: {count:4d} frames ({pct:5.1f}%) - Avg persons: {avg_persons:.1f}")
    print()
    
    # Detection distribution
    print(f"👥 DETECTION DISTRIBUTION")
    person_counts = defaultdict(int)
    for f in frames:
        person_counts[f['persons']] += 1
    
    for count in sorted(person_counts.keys())[:10]:  # Show top 10
        num_frames = person_counts[count]
        pct = num_frames * 100 / len(frames)
        print(f"  {count:2d} persons: {num_frames:4d} frames ({pct:5.1f}%)")
    print()
    
    # Best/worst frames
    print(f"🏆 BEST/WORST FRAMES")
    fastest = min(frames, key=lambda f: f['time_ms'])
    slowest = max(frames, key=lambda f: f['time_ms'])
    most_persons = max(frames, key=lambda f: f['persons'])
    
    print(f"  Fastest: {fastest['time_ms']:.1f}ms ({fastest['fps']:.2f} FPS) - {fastest['persons']} persons")
    print(f"  Slowest: {slowest['time_ms']:.1f}ms ({slowest['fps']:.2f} FPS) - {slowest['persons']} persons")
    print(f"  Most Detections: {most_persons['persons']} persons in {most_persons['time_ms']:.1f}ms")
    print()
    
    # Sample individual frames (every 10th frame)
    print(f"📋 SAMPLE FRAMES (showing every 10th)")
    print(f"  {'Frame':<8} {'Objects':<12} {'Time (ms)':<12} {'Accuracy':<12} {'FPS':<8}")
    print(f"  {'-'*60}")
    for i, f in enumerate(frames[::10][:20]):  # Show max 20 samples
        frame_num = i * 10
        obj_str = f"{f['persons']}p, {f['faces']}f"
        print(f"  {frame_num:<8d} {obj_str:<12} {f['time_ms']:<12.1f} {f['accuracy']:<12.3f} {f['fps']:<8.2f}")
    print()
    
    print("=" * 80)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_metrics.py <log_file>")
        sys.exit(1)
    
    log_file = sys.argv[1]
    frames = parse_log(log_file)
    analyze_frames(frames)
