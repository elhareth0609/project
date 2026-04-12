#!/usr/bin/env python3
"""
Kaggle Optimized Training Script - للتدريب على Kaggle
العديد من التحسينات الموصى بها طُبّقت
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║         🚀 IMPROVED YOLOV8 TRAINING - المحسّنة المراجعة 2026                   ║
║                                                                            ║
║                    تحسينات الأداء الرئيسية المطبقة:                          ║
║                                                                            ║
║  ✅ YOLOv8m بدلاً من YOLOv8n (25M vs 3M parameters)                         ║
║  ✅ 100 epochs بدلاً من 50 (تدريب أطول)                                    ║
║  ✅ imgsz 1280 بدلاً من 640 (دقة أعلى)                                     ║
║  ✅ patience=0 (تدريب كامل بدون توقف مبكر)                                ║
║  ✅ Data augmentation محسّن (rotations, translations, etc)                 ║
║  ✅ Learning rate محسّن (0.001 → 0.0001)                                   ║
║                                                                            ║
║  🎯 النتائج المتوقعة:                                                       ║
║  - Human-Footprint mAP@50: 0.50-0.65 (from 0.33)                          ║
║  - Vehicle mAP@50: 0.40-0.55 (from 0.17)                                  ║
║  - تحسن عام: +50% إلى +200%                                               ║
║                                                                            ║
║  ⏱️ الوقت المتوقع: 3-5 ساعات على Kaggle GPU                                ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

""")

class KaggleOptimizedPipeline:
    def __init__(self):
        self.project_root = Path("/kaggle/working/project")
        self.start_time = datetime.now()
        
    def print_config(self):
        """طباعة تكوين التدريب"""
        print("\n" + "="*80)
        print("TRAINING CONFIGURATION")
        print("="*80)
        print("""
📊 Model Architecture:
   - Base Model: YOLOv8m (Medium) - 25M parameters
   - Previous: YOLOv8n (Nano) - 3M parameters
   - Improvement: +733% more capacity ⬆️

🎓 Training Parameters:
   - Epochs: 100 (was 50) - double training duration
   - Image Size: 1280x1280 (was 640x640) - 4x resolution
   - Batch Size: 8 (reduced for medium model)
   - Learning Rate: 0.001-0.0001 (optimized)
   - Early Stopping: DISABLED (full epochs)

🎨 Data Augmentation:
   - Rotation: ±15° (was ±10°)
   - Translation: ±15% (was ±10%)
   - Scale: ±60% (was ±50%)
   - Perspective: ±10% (was 0%)
   - HSV: Full range (color variations)

📈 Expected Performance Gains:
   - mAP@50: +50% to +100% improvement expected
   - Recall: +100% to +200% improvement expected
   - Overall F1-Score: +50% to +150% improvement expected
        """)
        print("="*80 + "\n")
    
    def run_training(self):
        """تشغيل التدريب"""
        print("\n⏱️ Starting Optimized Training Pipeline...")
        print(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        scripts = [
            ('train_models.py', '🚀 Training Models'),
            ('evaluate_models.py', '📊 Evaluating & Computing Metrics'),
            ('visualize_results.py', '📈 Generating Visualizations'),
        ]
        
        results = {}
        
        for script, description in scripts:
            script_path = self.project_root / script
            
            if not script_path.exists():
                print(f"❌ {description} - FAILED (Script not found)")
                results[script] = False
                continue
            
            print(f"\n{'='*80}")
            print(f"{description}")
            print(f"{'='*80}\n")
            
            try:
                result = subprocess.run(
                    [sys.executable, str(script_path)],
                    cwd=str(self.project_root),
                    capture_output=False,
                    text=True
                )
                
                if result.returncode == 0:
                    print(f"\n✅ {description} - COMPLETED")
                    results[script] = True
                else:
                    print(f"\n❌ {description} - FAILED")
                    results[script] = False
                    
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                results[script] = False
        
        return results
    
    def print_summary(self, results):
        """طباعة الملخص"""
        elapsed = datetime.now() - self.start_time
        
        print("\n" + "="*80)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*80)
        
        for script, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"{script}: {status}")
        
        print(f"\nTotal Time: {elapsed}")
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if all(results.values()):
            print("\n" + "="*80)
            print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*80)
            print("""
📊 Results Location:
   ├── training_results/models/              (trained models)
   ├── training_results/metrics/             (validation metrics)
   │   ├── validation_metrics.json
   │   └── validation_metrics.csv
   ├── training_results/visualizations/      (charts & reports)
   │   ├── 01_metrics_comparison.png
   │   ├── 02_*_radar.png
   │   ├── 03_metrics_heatmap.png
   │   └── METRICS_REPORT.txt
   └── training_results/training_log.txt

📈 Expected Improvements:
   Human-Footprint:
   ├── mAP@50: 0.33 → 0.50-0.65 ⬆️
   ├── Recall: 0.31 → 0.60-0.75 ⬆️
   └── F1-Score: 0.45 → 0.60-0.70 ⬆️

   Vehicle:
   ├── mAP@50: 0.17 → 0.40-0.55 ⬆️
   ├── Recall: 0.19 → 0.50-0.65 ⬆️
   └── F1-Score: 0.26 → 0.50-0.60 ⬆️

🚀 Next Steps:
   1. Review metrics in training_results/metrics/
   2. Check visualizations for insights
   3. Use best.pt models for inference
   4. Share results with your team! 🎯
            """)
        else:
            print("\n❌ Some steps failed. Check logs for details.")
        
        print("="*80 + "\n")


def main():
    pipeline = KaggleOptimizedPipeline()
    pipeline.print_config()
    results = pipeline.run_training()
    pipeline.print_summary(results)


if __name__ == '__main__':
    main()
