#!/usr/bin/env python3
"""
Log Analyzer and Sequence Diagram Generator
Analyzes test logs and generates sequence diagrams
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import csv


class LogAnalyzer:
    """Analyze test logs and generate reports"""
    
    def __init__(self, logs_dir: str, reports_dir: str):
        self.logs_dir = Path(logs_dir)
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True, parents=True)
        
    def find_test_runs(self) -> List[str]:
        """Find all test run IDs"""
        run_ids = set()
        for log_file in self.logs_dir.glob("test_run_*_master.log"):
            run_id = log_file.stem.replace("_master", "")
            run_ids.add(run_id)
        return sorted(run_ids)
    
    def parse_master_log(self, run_id: str) -> Dict:
        """Parse master log file"""
        master_log = self.logs_dir / f"{run_id}_master.log"
        if not master_log.exists():
            return {}
        
        data = {
            'run_id': run_id,
            'timestamp': None,
            'input_file': None,
            'tests': []
        }
        
        with open(master_log, 'r') as f:
            content = f.read()
            
            # Extract metadata
            timestamp_match = re.search(r'Started at (.+)', content)
            if timestamp_match:
                data['timestamp'] = timestamp_match.group(1)
            
            input_match = re.search(r'Input file: (.+)', content)
            if input_match:
                data['input_file'] = input_match.group(1)
            
            # Extract test results
            test_pattern = r'Testing: (.+?) \((.+?), (.+?)\)'
            status_pattern = r'(✓ PASSED|✗ FAILED)'
            
            tests = re.finditer(test_pattern, content)
            for test in tests:
                test_info = {
                    'model': test.group(1),
                    'type': test.group(2),
                    'pipeline': test.group(3)
                }
                data['tests'].append(test_info)
        
        return data
    
    def parse_csv_results(self, run_id: str) -> List[Dict]:
        """Parse CSV results file"""
        csv_file = self.reports_dir / f"{run_id}_results.csv"
        if not csv_file.exists():
            return []
        
        results = []
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(row)
        
        return results
    
    def generate_sequence_diagram(self, run_id: str, results: List[Dict]):
        """Generate Mermaid sequence diagram"""
        diagram_path = self.reports_dir / f"{run_id}_sequence_diagram.md"
        
        # Separate by pipeline type
        sequential_tests = [r for r in results if r['Pipeline'] == 'sequential' and r['Status'] == 'PASS']
        parallel_tests = [r for r in results if r['Pipeline'] == 'parallel' and r['Status'] == 'PASS']
        
        diagram = f"""# Sequence Diagrams for Test Run: {run_id}

## Sequential Pipeline Flow

```mermaid
sequenceDiagram
    participant User
    participant Main
    participant Pipeline as Sequential Pipeline
    participant PersonDet as Person Detector
    participant FaceDet as Face Detector
    participant Metrics
    participant Output
    
    User->>Main: Input Image/Video
    Main->>Pipeline: create_pipeline(sequential)
    Pipeline->>PersonDet: Initialize
    Pipeline->>FaceDet: Initialize
    Pipeline->>Metrics: Initialize Tracker
    
    Main->>Pipeline: process_image(image)
    
    activate Pipeline
    Pipeline->>Metrics: start_processing()
    
    Pipeline->>PersonDet: detect(full_image)
    activate PersonDet
    PersonDet-->>Pipeline: [Person Detections]
    deactivate PersonDet
    
    loop For each person
        Pipeline->>Pipeline: crop_person_roi()
        Pipeline->>Output: save_person_crop()
        
        Pipeline->>FaceDet: detect(person_crop)
        activate FaceDet
        FaceDet-->>Pipeline: [Face Detections]
        deactivate FaceDet
        
        loop For each face
            Pipeline->>Output: save_face_crop()
        end
        
        Pipeline->>Metrics: add_frame_metrics()
    end
    
    Pipeline->>Metrics: end_processing()
    Pipeline->>Metrics: get_summary()
    Metrics-->>Pipeline: Performance Stats
    
    Pipeline-->>Main: (person_count, face_count)
    deactivate Pipeline
    
    Main->>Output: save_summary()
    Main->>User: Results + Metrics
```

### Sequential Pipeline Test Results

| Model | Type | Size (MB) | Time (s) | FPS | Detections |
|-------|------|-----------|----------|-----|------------|
"""
        
        for test in sequential_tests[:10]:  # Top 10
            diagram += f"| {test['Model']} | {test['Type']} | {test['Size_MB']} | {test['Processing_Time_s']} | {test['FPS']} | P:{test['Persons_Detected']} F:{test['Faces_Detected']} |\n"
        
        diagram += """

---

## Parallel Pipeline Flow

```mermaid
sequenceDiagram
    participant User
    participant Main
    participant Pipeline as Parallel Pipeline
    participant Thread1 as Thread 1
    participant Thread2 as Thread 2
    participant PersonDet as Person Detector
    participant FaceDet as Face Detector
    participant Metrics
    participant Output
    
    User->>Main: Input Image/Video
    Main->>Pipeline: create_pipeline(parallel)
    Pipeline->>PersonDet: Initialize
    Pipeline->>FaceDet: Initialize
    Pipeline->>Metrics: Initialize Tracker
    
    Main->>Pipeline: process_image(image)
    
    activate Pipeline
    Pipeline->>Metrics: start_processing()
    Pipeline->>Thread1: Start Person Detection
    Pipeline->>Thread2: Start Face Detection
    
    par Parallel Execution
        Thread1->>PersonDet: detect(full_image)
        activate PersonDet
        PersonDet-->>Thread1: [Person Detections]
        deactivate PersonDet
        Thread1->>Output: save_person_crops()
    and
        Thread2->>FaceDet: detect(full_image)
        activate FaceDet
        FaceDet-->>Thread2: [Face Detections]
        deactivate FaceDet
        Thread2->>Output: save_face_crops()
    end
    
    Thread1-->>Pipeline: Person Results
    Thread2-->>Pipeline: Face Results
    
    Pipeline->>Metrics: add_frame_metrics()
    Pipeline->>Metrics: end_processing()
    Pipeline->>Metrics: get_summary()
    Metrics-->>Pipeline: Performance Stats
    
    Pipeline-->>Main: (person_count, face_count)
    deactivate Pipeline
    
    Main->>Output: save_summary()
    Main->>User: Results + Metrics
```

### Parallel Pipeline Test Results

| Model | Type | Size (MB) | Time (s) | FPS | Detections |
|-------|------|-----------|----------|-----|------------|
"""
        
        for test in parallel_tests[:10]:  # Top 10
            diagram += f"| {test['Model']} | {test['Type']} | {test['Size_MB']} | {test['Processing_Time_s']} | {test['FPS']} | P:{test['Persons_Detected']} F:{test['Faces_Detected']} |\n"
        
        diagram += """

---

## Detailed Processing Flow

```mermaid
flowchart TD
    Start([Start Processing]) --> CheckInput{Input Type?}
    
    CheckInput -->|Image| LoadImage[Load Image]
    CheckInput -->|Video| LoadVideo[Load Video Frame]
    
    LoadImage --> Resize{Size > MAX?}
    LoadVideo --> Resize
    
    Resize -->|Yes| ResizeOp[Resize Image]
    Resize -->|No| PipelineSelect
    ResizeOp --> PipelineSelect
    
    PipelineSelect{Pipeline Mode?}
    
    PipelineSelect -->|Sequential| SeqStart[Sequential Pipeline]
    PipelineSelect -->|Parallel| ParStart[Parallel Pipeline]
    
    SeqStart --> DetectPerson[Detect Persons]
    DetectPerson --> PersonLoop{For Each Person}
    PersonLoop -->|Yes| CropPerson[Crop Person ROI]
    CropPerson --> SavePerson[Save Person Crop]
    SavePerson --> DetectFace[Detect Faces in Crop]
    DetectFace --> FaceLoop{For Each Face}
    FaceLoop -->|Yes| SaveFace[Save Face Crop]
    SaveFace --> FaceLoop
    FaceLoop -->|No| PersonLoop
    PersonLoop -->|No| CalcMetrics
    
    ParStart --> ThreadSplit[Split into Threads]
    ThreadSplit --> T1[Thread 1: Person Detection]
    ThreadSplit --> T2[Thread 2: Face Detection]
    T1 --> T1Save[Save Person Crops]
    T2 --> T2Save[Save Face Crops]
    T1Save --> ThreadJoin[Join Threads]
    T2Save --> ThreadJoin
    ThreadJoin --> CalcMetrics
    
    CalcMetrics[Calculate Metrics] --> SaveResults[Save Results]
    SaveResults --> End([End Processing])
```

---

## Model Loading Flow

```mermaid
flowchart TD
    Init([Initialize Detector]) --> CheckPath{Model Path<br/>Exists?}
    
    CheckPath -->|No| UseFallback[Use Fallback Detector]
    CheckPath -->|Yes| LoadONNX[Load ONNX Model]
    
    LoadONNX --> LoadSuccess{Load<br/>Success?}
    
    LoadSuccess -->|Yes| CreateSession[Create ONNX Session]
    LoadSuccess -->|No| UseFallback
    
    CreateSession --> SetProvider[Set CPU Provider]
    SetProvider --> Ready([Model Ready])
    
    UseFallback --> CheckType{Detector<br/>Type?}
    CheckType -->|Person| HOG[Initialize HOG + SVM]
    CheckType -->|Face| Cascade[Initialize Haar Cascade]
    
    HOG --> Ready
    Cascade --> Ready
```

---

## Performance Metrics Collection

```mermaid
sequenceDiagram
    participant Pipeline
    participant Metrics as MetricsTracker
    participant Storage
    
    Pipeline->>Metrics: start_processing()
    activate Metrics
    Note over Metrics: Record start_time
    
    loop For each detection
        Pipeline->>Metrics: add_frame_metrics(fps, accuracy, time)
        Metrics->>Storage: Append to lists
    end
    
    Pipeline->>Metrics: end_processing()
    Note over Metrics: Calculate total_time
    
    Pipeline->>Metrics: get_summary()
    Metrics->>Metrics: Calculate statistics
    Note over Metrics: - Min/Max/Avg FPS<br/>- Min/Max/Avg Accuracy<br/>- Avg Processing Time
    Metrics-->>Pipeline: Summary Dict
    
    Pipeline->>Metrics: save_to_file(path)
    Metrics->>Storage: Write summary.txt
    
    Pipeline->>Metrics: print_summary()
    Metrics->>Storage: Output to console
    deactivate Metrics
```

---

## Generated by Log Analyzer
**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Run ID:** {run_id}
"""
        
        with open(diagram_path, 'w') as f:
            f.write(diagram)
        
        return diagram_path
    
    def generate_performance_report(self, run_id: str, results: List[Dict]):
        """Generate detailed performance analysis report"""
        report_path = self.reports_dir / f"{run_id}_performance_analysis.md"
        
        # Calculate statistics
        total_tests = len(results)
        passed = sum(1 for r in results if r['Status'] == 'PASS')
        failed = total_tests - passed
        
        # Group by model type
        person_models = [r for r in results if r['Type'] == 'person' and r['Status'] == 'PASS']
        face_models = [r for r in results if r['Type'] == 'face' and r['Status'] == 'PASS']
        
        # Sort by different metrics
        fastest_models = sorted(results, key=lambda x: float(x['FPS']) if x['FPS'] else 0, reverse=True)[:5]
        smallest_models = sorted(results, key=lambda x: float(x['Size_MB']) if x['Size_MB'] else 999)[:5]
        
        report = f"""# Performance Analysis Report

**Run ID:** {run_id}  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

- **Total Tests:** {total_tests}
- **Passed:** {passed} ({passed/total_tests*100:.1f}%)
- **Failed:** {failed} ({failed/total_tests*100:.1f}%)
- **Person Detection Models:** {len([r for r in results if r['Type'] == 'person'])}
- **Face Detection Models:** {len([r for r in results if r['Type'] == 'face'])}

---

## Top Performers

### 🚀 Fastest Models (by FPS)

| Rank | Model | Type | Pipeline | FPS | Time (s) |
|------|-------|------|----------|-----|----------|
"""
        
        for i, model in enumerate(fastest_models, 1):
            if model['Status'] == 'PASS':
                report += f"| {i} | {model['Model']} | {model['Type']} | {model['Pipeline']} | {model['FPS']} | {model['Processing_Time_s']} |\n"
        
        report += """

### 💾 Smallest Models (by size)

| Rank | Model | Type | Size (MB) | FPS |
|------|-------|------|-----------|-----|
"""
        
        for i, model in enumerate(smallest_models, 1):
            if model['Status'] == 'PASS':
                report += f"| {i} | {model['Model']} | {model['Type']} | {model['Size_MB']} | {model['FPS']} |\n"
        
        report += """

---

## Person Detection Analysis

### Model Comparison

| Model | Size (MB) | Sequential FPS | Parallel FPS | Detections | Winner |
|-------|-----------|----------------|--------------|------------|--------|
"""
        
        # Group person models by base name
        person_groups = {}
        for model in person_models:
            base_name = model['Model'].split('-')[0]
            if base_name not in person_groups:
                person_groups[base_name] = []
            person_groups[base_name].append(model)
        
        for base_name, models in person_groups.items():
            seq_model = next((m for m in models if m['Pipeline'] == 'sequential'), None)
            par_model = next((m for m in models if m['Pipeline'] == 'parallel'), None)
            
            if seq_model and par_model:
                seq_fps = float(seq_model['FPS']) if seq_model['FPS'] else 0
                par_fps = float(par_model['FPS']) if par_model['FPS'] else 0
                winner = "Parallel" if par_fps > seq_fps else "Sequential"
                
                report += f"| {base_name} | {seq_model['Size_MB']} | {seq_fps:.2f} | {par_fps:.2f} | {seq_model['Persons_Detected']} | {winner} |\n"
        
        report += """

### Recommendations for Person Detection

"""
        
        if person_models:
            fastest_person = max(person_models, key=lambda x: float(x['FPS']) if x['FPS'] else 0)
            smallest_person = min(person_models, key=lambda x: float(x['Size_MB']) if x['Size_MB'] else 999)
            
            report += f"""
**For Speed:** Use `{fastest_person['Model']}` with `{fastest_person['Pipeline']}` pipeline
- FPS: {fastest_person['FPS']}
- Processing Time: {fastest_person['Processing_Time_s']}s

**For Memory:** Use `{smallest_person['Model']}`
- Size: {smallest_person['Size_MB']} MB
- FPS: {smallest_person['FPS']}
"""
        
        report += """

---

## Face Detection Analysis

### Model Comparison

| Model | Size (MB) | Sequential FPS | Parallel FPS | Detections | Winner |
|-------|-----------|----------------|--------------|------------|--------|
"""
        
        # Group face models
        face_groups = {}
        for model in face_models:
            base_name = model['Model'].split('-')[0]
            if base_name not in face_groups:
                face_groups[base_name] = []
            face_groups[base_name].append(model)
        
        for base_name, models in face_groups.items():
            seq_model = next((m for m in models if m['Pipeline'] == 'sequential'), None)
            par_model = next((m for m in models if m['Pipeline'] == 'parallel'), None)
            
            if seq_model and par_model:
                seq_fps = float(seq_model['FPS']) if seq_model['FPS'] else 0
                par_fps = float(par_model['FPS']) if par_model['FPS'] else 0
                winner = "Parallel" if par_fps > seq_fps else "Sequential"
                
                report += f"| {base_name} | {seq_model['Size_MB']} | {seq_fps:.2f} | {par_fps:.2f} | {seq_model['Faces_Detected']} | {winner} |\n"
        
        report += """

### Recommendations for Face Detection

"""
        
        if face_models:
            fastest_face = max(face_models, key=lambda x: float(x['FPS']) if x['FPS'] else 0)
            smallest_face = min(face_models, key=lambda x: float(x['Size_MB']) if x['Size_MB'] else 999)
            
            report += f"""
**For Speed:** Use `{fastest_face['Model']}` with `{fastest_face['Pipeline']}` pipeline
- FPS: {fastest_face['FPS']}
- Processing Time: {fastest_face['Processing_Time_s']}s

**For Memory:** Use `{smallest_face['Model']}`
- Size: {smallest_face['Size_MB']} MB
- FPS: {smallest_face['FPS']}
"""
        
        report += """

---

## Pipeline Comparison

### Sequential vs Parallel Performance

"""
        
        seq_results = [r for r in results if r['Pipeline'] == 'sequential' and r['Status'] == 'PASS']
        par_results = [r for r in results if r['Pipeline'] == 'parallel' and r['Status'] == 'PASS']
        
        if seq_results:
            avg_seq_fps = sum(float(r['FPS']) for r in seq_results if r['FPS']) / len(seq_results)
            report += f"\n**Sequential Pipeline:**\n"
            report += f"- Average FPS: {avg_seq_fps:.2f}\n"
            report += f"- Total Tests: {len(seq_results)}\n"
        
        if par_results:
            avg_par_fps = sum(float(r['FPS']) for r in par_results if r['FPS']) / len(par_results)
            report += f"\n**Parallel Pipeline:**\n"
            report += f"- Average FPS: {avg_par_fps:.2f}\n"
            report += f"- Total Tests: {len(par_results)}\n"
        
        report += """

### When to Use Each Pipeline

**Use Sequential Pipeline when:**
- Multiple persons in the scene
- Face-person association is needed
- Memory is limited
- Accuracy is more important than speed

**Use Parallel Pipeline when:**
- Single person scenarios
- Speed is critical
- Faces might be outside person bounding boxes
- Independent detection is acceptable

---

## Optimization Recommendations

### For Real-Time Performance (>5 FPS)
```bash
# Use fastest models with parallel pipeline
python main.py --input input/video.mp4 \\
    --person-model models/NanoDet/object_detection_nanodet_2022nov_int8.onnx \\
    --face-model models/YuNet/face_detection_yunet_2023mar_int8.onnx \\
    --pipeline parallel
```

### For High Accuracy
```bash
# Use larger models with sequential pipeline
python main.py --input input/photo.jpg \\
    --person-model models/RF-DETR-Nano/model.onnx \\
    --face-model models/YOLOv8-Face/yolov8n-face.onnx \\
    --pipeline sequential \\
    --person-threshold 0.6 \\
    --face-threshold 0.6
```

### For Low Memory (< 1GB)
```bash
# Use quantized models
python main.py --input input/photo.jpg \\
    --person-model models/NanoDet/object_detection_nanodet_2022nov_int8.onnx \\
    --face-model models/YuNet/face_detection_yunet_2023mar_int8.onnx
```

---

## Failed Tests Analysis

"""
        
        failed_tests = [r for r in results if r['Status'] == 'FAIL']
        if failed_tests:
            report += f"\n**Total Failed:** {len(failed_tests)}\n\n"
            report += "| Model | Type | Pipeline | Notes |\n"
            report += "|-------|------|----------|-------|\n"
            for test in failed_tests:
                report += f"| {test['Model']} | {test['Type']} | {test['Pipeline']} | {test['Notes']} |\n"
        else:
            report += "\n✅ All tests passed successfully!\n"
        
        report += """

---

## Conclusion

Based on the test results, we recommend:

1. **Default Configuration:** Use models that balance speed and accuracy
2. **Monitor Performance:** Track FPS and adjust as needed
3. **Test with Real Data:** Results may vary with different input types
4. **Consider Trade-offs:** Speed vs Accuracy vs Memory

For detailed logs, check:
- Individual test logs: `logs/{run_id}_*.log`
- Master log: `logs/{run_id}_master.log`
- Sequence diagrams: `docs/reports/{run_id}_sequence_diagram.md`

---

**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        return report_path
    
    def analyze_latest_run(self):
        """Analyze the most recent test run"""
        run_ids = self.find_test_runs()
        if not run_ids:
            print("No test runs found!")
            return
        
        latest_run = run_ids[-1]
        print(f"Analyzing latest test run: {latest_run}")
        
        # Parse results
        results = self.parse_csv_results(latest_run)
        if not results:
            print(f"No results found for {latest_run}")
            return
        
        # Generate reports
        print("\nGenerating reports...")
        
        diagram_path = self.generate_sequence_diagram(latest_run, results)
        print(f"✓ Sequence diagram: {diagram_path}")
        
        report_path = self.generate_performance_report(latest_run, results)
        print(f"✓ Performance report: {report_path}")
        
        print(f"\nAnalysis complete! Check {self.reports_dir}/")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze test logs and generate reports")
    parser.add_argument('--logs-dir', default='logs', help='Logs directory')
    parser.add_argument('--reports-dir', default='docs/reports', help='Reports output directory')
    parser.add_argument('--run-id', help='Specific run ID to analyze (default: latest)')
    
    args = parser.parse_args()
    
    analyzer = LogAnalyzer(args.logs_dir, args.reports_dir)
    
    if args.run_id:
        results = analyzer.parse_csv_results(args.run_id)
        if results:
            analyzer.generate_sequence_diagram(args.run_id, results)
            analyzer.generate_performance_report(args.run_id, results)
            print(f"Analysis complete for {args.run_id}")
        else:
            print(f"No results found for {args.run_id}")
    else:
        analyzer.analyze_latest_run()


if __name__ == "__main__":
    main()
