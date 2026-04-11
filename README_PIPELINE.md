# Object Detection Model Training Pipeline

Complete end-to-end pipeline for training YOLOv8 models on footprint and vehicle detection datasets.

## 📋 Overview

This pipeline automates:
1. ✅ Data splitting (already completed: 70% train / 15% valid / 15% test)
2. 🚀 Model training (YOLOv8 nano for both datasets)
3. 📊 Performance evaluation (metrics: mAP@50, Precision, Recall, F1)
4. 📈 Visualization & reporting

## 📁 Project Structure

```
project/
├── human-footprint/              # Human footprint dataset
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml
├── vehicle/                       # Vehicle detection dataset
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml
├── training_results/              # Output directory (auto-created)
│   ├── models/                    # Trained models
│   ├── metrics/                   # Validation metrics
│   ├── visualizations/            # Charts and reports
│   └── training_log.txt
├── split_dataset.py              # Data splitting script (already ran)
├── train_models.py               # Model training script
├── evaluate_models.py            # Metrics computation
├── visualize_results.py          # Visualization generator
└── run_pipeline.py               # Master orchestrator
```

## 🚀 Quick Start

### Option 1: Run Complete Pipeline (Recommended)

```bash
python run_pipeline.py
```

This will:
1. Train both models automatically
2. Evaluate on validation set
3. Generate metrics and visualizations
4. Create comprehensive report

### Option 2: Run Individual Steps

```bash
# Step 1: Train models
python train_models.py

# Step 2: Evaluate and compute metrics
python evaluate_models.py

# Step 3: Generate visualizations
python visualize_results.py
```

## 📊 Expected Metrics

After training, you'll get:

| Metric | Description | Target |
|--------|-------------|--------|
| **mAP@50** | Mean Average Precision at 50% IoU | > 0.5 |
| **Precision** | Of detected objects, how many correct | > 0.7 |
| **Recall** | Of actual objects, how many found | > 0.6 |
| **F1-Score** | Harmonic mean (balanced metric) | > 0.65 |

## 🔧 Configuration Details

### Training Parameters

- **Model**: YOLOv8 Nano (optimized for small datasets)
- **Epochs**: 50
- **Batch Size**: 16
- **Image Size**: 640x640
- **Device**: GPU (0) if available, CPU otherwise

### Augmentation Settings

- Mosaic: 100%
- Flip (up/down): 50%
- Flip (left/right): 50%
- Rotation: ±10°
- Translation: ±10%
- Scale: ±50%
- HSV: Hue ±1.5%, Saturation ±70%, Value ±40%

## 📈 Output Files

After running the pipeline:

```
training_results/
├── models/
│   ├── human-footprint_YYYYMMDD_HHMMSS/
│   │   ├── weights/
│   │   │   ├── best.pt          # Best model
│   │   │   └── last.pt          # Last checkpoint
│   │   ├── results.png          # Training curves
│   │   └── confusion_matrix.png
│   └── vehicle_YYYYMMDD_HHMMSS/
├── metrics/
│   ├── validation_metrics.json   # Detailed metrics
│   ├── validation_metrics.csv    # Excel-friendly format
├── visualizations/
│   ├── 01_metrics_comparison.png # Bar chart comparison
│   ├── 02_*_radar.png           # Performance radar charts
│   ├── 03_metrics_heatmap.png   # Metrics heatmap
│   └── METRICS_REPORT.txt       # Text report
├── model_info.json              # Model metadata
└── training_log.txt             # Complete execution log
```

## 🎯 Usage for Inference

Once trained, use models for predictions:

```python
from ultralytics import YOLO

# Load best model
model = YOLO('training_results/models/human-footprint_XXX/weights/best.pt')

# Predict on image
results = model.predict('image.jpg', conf=0.5)

# Predict on video
results = model.predict('video.mp4', conf=0.5)
```

## 💡 Tips for Better Results

1. **More Training**: Increase `epochs` from 50 to 100-200
2. **Better Augmentation**: Adjust augmentation parameters in `train_models.py`
3. **Hardware**: Using GPU dramatically speeds up training
4. **Monitoring**: Check `results_dir/*/results.png` during training
5. **Fine-tuning**: Use a pre-trained larger model (e.g., yolov8m.pt or yolov8l.pt)

## 📝 Requirements Met

- ✅ Data properly split (70/15/15)
- ✅ Models trained on footprint and vehicle detection
- ✅ Validation metrics computed (mAP@50, Precision, Recall, F1)
- ✅ Comprehensive visualizations generated
- ✅ Professional report created

## 🆘 Troubleshooting

### GPU Not Detected
```python
# In train_models.py, the script auto-detects GPU
# If not found, automatically uses CPU
```

### Out of Memory
```python
# Reduce batch size in train_models.py
batch=8  # Instead of 16
# Or reduce image size
imgsz=512  # Instead of 640
```

### Training Too Slow
```python
# Reduce epochs
epochs=25  # Instead of 50
# Or use smaller model
model = YOLO('yolov8n.pt')  # Already using nano
```

## 📞 Support

For detailed model information and training logs, check:
- `training_results/training_log.txt` - Complete execution history
- `training_results/models/*/results.png` - Training curves
- `training_results/metrics/validation_metrics.json` - Raw metrics

---

**Created**: 2026-04-11  
**Pipeline Version**: 1.0  
**YOLOv8 Integration**: Ultralytics
