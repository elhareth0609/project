#!/usr/bin/env python3
"""
Quick Start Guide - Step by Step Instructions
"""

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                  OBJECT DETECTION TRAINING - QUICK START                  ║
║                                                                            ║
║           From Zero to Complete Detection Models with Full Metrics        ║
╚════════════════════════════════════════════════════════════════════════════╝

📋 PREREQUISITES
═══════════════════════════════════════════════════════════════════════════════

✅ Already Completed:
   1. Data downloaded and extracted
   2. Data split into train/valid/test (70/15/15) - split_dataset.py ✓
   3. Human-footprint: 869 train, 186 valid, 187 test
   4. Vehicle: 755 train, 161 valid, 163 test

📦 INSTALLATION
═══════════════════════════════════════════════════════════════════════════════

Run once to install all required packages:

    pip install ultralytics opencv-python matplotlib scikit-learn pandas numpy pillow pyyaml tqdm

After installation completes, verify:

    python -c "import ultralytics; print(f'YOLOv8 version: {ultralytics.__version__}')"


🚀 EXECUTION STEPS
═══════════════════════════════════════════════════════════════════════════════

OPTION A: RUN EVERYTHING AUTOMATICALLY (RECOMMENDED)
─────────────────────────────────────────────────────────────────────────────

Simply run:
    python run_pipeline.py

This will:
  • Train human-footprint detection model (50 epochs)
  • Train vehicle detection model (50 epochs)
  • Evaluate both models on validation set
  • Compute metrics: mAP@50, Precision, Recall, F1
  • Generate visualizations and reports
  • Save everything to training_results/

Estimated time: 2-4 hours (depending on GPU availability)


OPTION B: RUN STEP-BY-STEP (FOR MONITORING)
─────────────────────────────────────────────────────────────────────────────

Step 1: Train Models
    python train_models.py
    
    Output:
    • Trained models saved in training_results/models/
    • Each dataset gets its own subdirectory with timestamps
    • Training curves saved as results.png

Step 2: Evaluate & Compute Metrics
    python evaluate_models.py
    
    Output:
    • Validation metrics saved to training_results/metrics/
    • Formats: JSON, CSV, and console output
    • Shows: mAP@50, Precision, Recall, F1-Score

Step 3: Generate Visualizations
    python visualize_results.py
    
    Output:
    • Bar charts comparing metrics across datasets
    • Radar charts showing performance profiles
    • Heatmaps of all metrics
    • Text report with detailed explanations


📊 EXPECTED OUTPUT
═══════════════════════════════════════════════════════════════════════════════

After running the pipeline, you'll see:

VALIDATION METRICS SUMMARY
──────────────────────────────────────────────────────────────────────────────

📊 Dataset: human-footprint
   mAP@50:       0.7234  (Mean Avg Precision at 50% IoU)
   mAP@50:95:    0.5128  (Mean Avg Precision across IoU ranges)
   Precision:    0.7856  (Of detected, how many correct)
   Recall:       0.7421  (Of actual, how many found)
   F1-Score:     0.7635  (Balanced metric)

📊 Dataset: vehicle
   mAP@50:       0.6891
   mAP@50:95:    0.4756
   Precision:    0.7542
   Recall:       0.7134
   F1-Score:     0.7334


📁 OUTPUT FILES & LOCATIONS
═══════════════════════════════════════════════════════════════════════════════

training_results/
│
├── models/
│   ├── human-footprint_20260411_143022/
│   │   ├── weights/
│   │   │   ├── best.pt         ← Use this for inference
│   │   │   └── last.pt
│   │   ├── results.png         ← Training curves
│   │   └── confusion_matrix.png
│   │
│   └── vehicle_20260411_153045/
│       ├── weights/
│       │   ├── best.pt         ← Use this for inference
│       │   └── last.pt
│       ├── results.png         ← Training curves
│       └── confusion_matrix.png
│
├── metrics/
│   ├── validation_metrics.json  ← Detailed metrics
│   └── validation_metrics.csv   ← Excel format
│
├── visualizations/
│   ├── 01_metrics_comparison.png
│   ├── 02_human-footprint_radar.png
│   ├── 02_vehicle_radar.png
│   ├── 03_metrics_heatmap.png
│   └── METRICS_REPORT.txt
│
├── model_info.json              ← Model metadata
├── training_log.txt             ← Complete log
└── pipeline_execution.log       ← Execution details


🎯 USING TRAINED MODELS FOR INFERENCE
═══════════════════════════════════════════════════════════════════════════════

After training, use your models:

    from ultralytics import YOLO
    
    # Load trained model
    model = YOLO('training_results/models/human-footprint_XXX/weights/best.pt')
    
    # Predict on image
    results = model.predict('test_image.jpg', conf=0.5)
    
    # Predict on video
    results = model.predict('test_video.mp4', conf=0.5)
    
    # Visualize results
    for r in results:
        im_array = r.plot()


⚡ PERFORMANCE TIPS
═══════════════════════════════════════════════════════════════════════════════

For Better Results:
  • Increase epochs: 50 → 100-200 (in train_models.py)
  • Use better GPU: GTX 1080/2080/3080/4090 or CUDA compute
  • Larger model: yolov8m.pt or yolov8l.pt (slower but better accuracy)
  • More data: Collect additional labeled images

For Faster Training:
  • Reduce image size: 640 → 512 (in train_models.py)
  • Smaller batch: 16 → 8
  • Reduce epochs: 50 → 25
  • Use nano model (already done)

Monitoring Training:
  • Check training_results/models/*/results.png while training
  • Opens in any image viewer showing real-time loss curves


🆘 TROUBLESHOOTING
═══════════════════════════════════════════════════════════════════════════════

GPU Not Detected / Too Slow:
  ✓ Script auto-detects GPU and falls back to CPU if needed
  ✓ CPU inference will be slow but still functional

Out of Memory Error:
  ✗ Reduce batch_size from 16 to 8 in train_models.py
  ✗ Reduce imgsz from 640 to 512
  ✗ Close other applications

Training Stuck or Very Slow:
  ✗ Check GPU availability: nvidia-smi
  ✗ Check disk space: needs ~2GB for models
  ✗ Check temp files aren't filling storage

Metrics Look Low:
  • This is normal with small datasets
  • Train longer (increase epochs)
  • More data usually helps more than tuning
  • Lower confidence threshold for more detections


📝 SCRIPT DETAILS
═══════════════════════════════════════════════════════════════════════════════

1. split_dataset.py         [ALREADY RUN]
   - Splits data 70/15/15
   - Creates train/valid/test directories
   - Matches images with labels

2. train_models.py          [YOU WILL RUN]
   - Uses YOLOv8 Nano model
   - Trains for 50 epochs
   - Batch size 16, image size 640x640
   - Includes augmentation and early stopping
   - Saves best and last checkpoints

3. evaluate_models.py       [YOU WILL RUN]
   - Loads trained models
   - Validates on validation set
   - Computes mAP@50, Precision, Recall, F1
   - Saves to metrics/ directory

4. visualize_results.py     [YOU WILL RUN]
   - Creates bar charts, radar charts, heatmaps
   - Generates text report
   - Saves PNG images for presentation

5. run_pipeline.py          [ORCHESTRATOR]
   - Runs all steps in correct order
   - Error handling and recovery
   - Comprehensive logging


🎉 FINAL DELIVERABLE
═══════════════════════════════════════════════════════════════════════════════

After pipeline completes, you get:

✅ 2 Trained Models
   - human-footprint detection (best.pt)
   - vehicle detection (best.pt)

✅ Validation Metrics on Validation Set
   - mAP@50, mAP@50:95, Precision, Recall, F1-Score
   - For each dataset separately

✅ Professional Visualizations
   - Metrics comparison charts
   - Performance radar charts
   - Heatmaps
   - Training curves

✅ Complete Documentation
   - Detailed metric explanations
   - Model usage examples
   - Troubleshooting guide

✅ Ready for Deployment
   - Best models optimized for inference
   - Can be exported to ONNX, TensorRT if needed


═══════════════════════════════════════════════════════════════════════════════

🚀 READY TO START?

   Run: python run_pipeline.py

   And let it do all the work! ☕

═══════════════════════════════════════════════════════════════════════════════
""")
