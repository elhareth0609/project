# YOLO 11 Footprint Detection - Setup & Training Guide

## 📊 Dataset Overview
- **Animals Dataset**: 1,556 images
- **Human Dataset**: 1,242 images  
- **Total**: 2,798 images
- **Classes**: 2 (animal_footprint, human_footprint)
- **Split**: 70% train, 15% val, 15% test

---

## 🚀 STEP 1: Run Dataset Split (Local)

Run this locally to split and combine your datasets:

```bash
python split_dataset.py
```

This will:
✅ Split animals dataset into train/val/test  
✅ Split humain dataset into train/val/test  
✅ Combine both datasets into `dataset_combined` folder  
✅ Create proper `data.yaml` with 2 classes

**Expected Output:**
```
dataset_combined/
├── train/
│   ├── images/ (1,958 images)
│   └── labels/
├── val/
│   ├── images/ (419 images)
│   └── labels/
├── test/
│   ├── images/ (419 images)
│   └── labels/
└── data.yaml
```

---

## 🎓 STEP 2: Option A - Train on KAGGLE (Recommended)

### Method 1: Upload & Use Notebook

1. **Go to [Kaggle.com](https://kaggle.com)**

2. **Create New Dataset**
   - Click "Create" → "New Dataset"
   - Create a new dataset called "footprint-detection"
   
3. **Upload Data**
   - Upload the `dataset_combined` folder
   - Or upload `animals` and `humain` separately

4. **Create Kaggle Notebook**
   - New Notebook in Kaggle
   - Select your dataset as input
   - Copy-paste content from `kaggle_yolo11_notebook.py`
   - Run all cells

5. **Configure GPU (Important!)**
   - Notebook Settings → Accelerator → GPU
   - Select T4 GPU

### Method 2: Use Kaggle API (Advanced)

```bash
# Install Kaggle CLI
pip install kaggle

# Configure authentication (download from https://www.kaggle.com/settings)
# Place kaggle.json in ~/.kaggle/

# Upload dataset
kaggle datasets create -p ./dataset_combined -u

# Download trained model
kaggle competitions download -c <competition-name>
```

---

## 💻 STEP 3: Option B - Train Locally (Advanced)

### Prerequisites
```bash
pip install ultralytics torch torchvision pyyaml opencv-python numpy pandas
```

### Quick Start
```bash
python train_yolo11_kaggle.py
```

Or use Python directly:

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolo11m.pt')

# Train
results = model.train(
    data='dataset_combined/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    device=0  # GPU index
)

# Validate
metrics = model.val()

# Predict
predictions = model.predict(source='image.jpg', conf=0.25)
```

---

## 🧪 STEP 4: Validate Your Code

### Check Dataset Structure
```bash
# Windows PowerShell
ls -Recurse dataset_combined | Group-Object -Property Extension

# Count images
(ls dataset_combined/train/images).Count
```

### Verify YOLO Setup
```python
from ultralytics import YOLO
import torch

print(f"PyTorch: {torch.__version__}")
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"YOLO11 Available: {YOLO('yolo11m.pt')}")
```

---

## 📁 File Structure

```
project/
├── split_dataset.py              # ✅ Splits & combines datasets
├── train_yolo11_kaggle.py        # ✅ Local training script
├── kaggle_yolo11_notebook.py     # ✅ Copy to Kaggle notebook
├── animals/
│   ├── train/images/
│   ├── train/labels/
│   ├── valid/images/
│   ├── valid/labels/
│   ├── test/images/
│   └── test/labels/
├── humain/
│   ├── train/images/
│   ├── train/labels/
│   ├── valid/images/
│   ├── valid/labels/
│   ├── test/images/
│   └── test/labels/
└── dataset_combined/             # ✅ After running split_dataset.py
    ├── train/images/ & labels/
    ├── val/images/ & labels/
    ├── test/images/ & labels/
    └── data.yaml
```

---

## ✅ Code Validation Checklist

### 1. Dataset Preparation
- ✅ YOLO format (txt labels with normalized coordinates)
- ✅ Class IDs properly remapped (0: animal_footprint, 1: human_footprint)
- ✅ data.yaml has correct paths and class names
- ✅ All images have corresponding label files

### 2. Training Script
- ✅ Uses `ultralytics>=8.0.0`
- ✅ GPU auto-detection
- ✅ Proper augmentation parameters
- ✅ Early stopping (patience=10)
- ✅ Model checkpointing

### 3. Kaggle Compatibility
- ✅ No local file paths hardcoded
- ✅ Uses `/kaggle/input` for uploads
- ✅ Saves to `/kaggle/working`
- ✅ Works with T4/P100 GPU

---

## 🎯 Training Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | yolo11m | Medium model, good balance |
| Epochs | 50 | Usually converges by 30-40 |
| Batch Size | 16 | Adjust down if GPU OOM |
| Image Size | 640 | Standard YOLO size |
| Device | GPU (0) | Auto-detects CUDA |
| Optimizer | auto | Adaptive optimizer |
| LR0 | 0.01 | Initial learning rate |
| Patience | 10 | Early stopping patience |

### Model Size Options
- **nano** (n): Fastest, 40MB
- **small** (s): Fast, 50MB
- **medium** (m): ⭐ Recommended, 50MB
- **large** (l): Slower, 75MB
- **xlarge** (x): Slowest, 145MB

---

## 📊 Expected Results

With 2,798 images and proper training:
- **mAP@50**: 0.85-0.95 (85-95% accuracy)
- **Training Time**: ~30-60 min on Kaggle GPU
- **Model Size**: ~50MB (weights/best.pt)

---

## 🔍 Quick Diagnosis

### Issue: "FileNotFoundError: data.yaml"
```
Solution: Run split_dataset.py first
python split_dataset.py
```

### Issue: "CUDA out of memory"
```
Solution: Reduce batch size
batch=8  # Instead of 16
```

### Issue: "No module named ultralytics"
```
Solution: Install ultralytics
pip install ultralytics
```

### Issue: Images not loading
```
Check:
1. Image files exist in dataset_combined/train/images/
2. File extensions are .jpg or .png
3. Label files exist for each image
4. Paths in data.yaml are correct
```

---

## 📝 Next Steps

1. ✅ Run `split_dataset.py` to prepare data
2. ✅ Check dataset structure
3. ✅ Upload to Kaggle
4. ✅ Run training in Kaggle Notebook
5. ✅ Download best.pt model
6. ✅ Use model for inference

---

## 🎓 Use Trained Model

```python
from ultralytics import YOLO
from PIL import Image

# Load model
model = YOLO('path/to/best.pt')

# Single image prediction
results = model.predict(
    source='footprint.jpg',
    conf=0.25,  # Confidence threshold
    save=True   # Save predictions
)

# Batch predictions
results = model.predict(
    source='path/to/images/',
    batch=32,
    conf=0.25
)

# Get results
for r in results:
    print(f"Detections: {len(r.boxes)}")
    print(f"Class IDs: {r.boxes.cls}")
    print(f"Confidences: {r.boxes.conf}")
```

---

## 📚 References

- [YOLO Documentation](https://docs.ultralytics.com)
- [Kaggle Notebook Guide](https://www.kaggle.com/docs/notebooks)
- [PyTorch Installation](https://pytorch.org/get-started/locally/)

---

**⚠️ Important**: Make sure to:
- Have CUDA installed (for GPU training)
- Use proper YOLO label format (class_id x_center y_center width height)
- Test dataset locally before uploading to Kaggle
