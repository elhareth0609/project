# 🚀 YOLO 11 Kaggle Training - QUICK START

## ✅ YOUR DATASETS ARE READY!

**Status**: ✅ Split and Combined Successfully  
**Total Images**: 2,798  
**Train**: 1,958 (70%)  
**Validation**: 419 (15%)  
**Test**: 421 (15%)  
**Classes**: 2 (animal_footprint, human_footprint)

---

## 📋 HOW TO RUN ON KAGGLE

### Step 1: Upload Dataset to Kaggle

1. Go to [Kaggle.com](https://kaggle.com)
2. Click **"Create"** → **"New Dataset"**
3. Upload the `dataset_combined` folder
   - Or upload the `animals` and `humain` folders separately

### Step 2: Create Kaggle Notebook

1. Go to Kaggle
2. Click **"Create"** → **"New Notebook"**
3. Select your dataset as input
4. Copy-paste the code below OR use **kaggle_yolo11_notebook.py**

### Step 3: Enable GPU

⚠️ **IMPORTANT**: Before running!
- Click **"Session"** button (top-right)
- Select **"Accelerator"** → **"GPU"** (T4)
- Your notebook will restart with GPU

### Step 4: Run Training

Copy-paste this complete code into your Kaggle notebook:

```python
# ============================================================
# KAGGLE NOTEBOOK - YOLO11 TRAINING
# ============================================================

# Install ultralytics
!pip install -q ultralytics

import os
import shutil
import random
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch

print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ GPU: {torch.cuda.is_available()}")

# ============================================================
# SETUP PATHS
# ============================================================

working_dir = Path('/kaggle/working')
input_dir = Path('/kaggle/input')

# Find your dataset (uploaded folder)
# Adjust the path based on what you uploaded
dataset_dir = working_dir / 'dataset_combined'
dataset_dir.mkdir(exist_ok=True)

# If you uploaded dataset_combined directly:
if (input_dir / 'dataset-combined').exists():
    shutil.copytree(
        input_dir / 'dataset-combined', 
        dataset_dir, 
        dirs_exist_ok=True
    )
    print("✅ Dataset copied from input")

# If you uploaded animals and humain separately, use the split code below:
# (uncomment if needed)

# ============================================================
# LOAD DATA CONFIG
# ============================================================

if (dataset_dir / 'data.yaml').exists():
    with open(dataset_dir / 'data.yaml', 'r') as f:
        data = yaml.safe_load(f)
    
    print("\n" + "="*60)
    print("DATASET CONFIG")
    print("="*60)
    print(f"Classes: {data['nc']}")
    print(f"Names: {data['names']}")
    print(f"Train: {data['train']}")
    print(f"Val: {data['val']}")
    print(f"Test: {data['test']}")
    print("="*60)
else:
    print("⚠️ data.yaml not found. Make sure dataset is uploaded correctly.")

# ============================================================
# TRAIN YOLO11
# ============================================================

print("\n" + "="*60)
print("TRAINING YOLO11")
print("="*60 + "\n")

# Model options: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
model = YOLO('yolo11m.pt')

results = model.train(
    data=str(dataset_dir / 'data.yaml'),
    epochs=50,
    imgsz=640,
    batch=16,
    patience=10,
    save=True,
    device=0,  # GPU
    project='/kaggle/working',
    name='yolo11_footprint_detection',
    exist_ok=False,
    verbose=True,
    # Augmentation
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10,
    translate=0.1,
    scale=0.5,
    flipud=0.5,
    fliplr=0.5,
)

print("\n✅ Training complete!")

# ============================================================
# EVALUATE
# ============================================================

best_model_path = Path('/kaggle/working/yolo11_footprint_detection/weights/best.pt')

if best_model_path.exists():
    print("\nLoading best model for evaluation...")
    best_model = YOLO(str(best_model_path))
    
    # Validate
    val_results = best_model.val()
    
    print(f"\n📊 Validation Results:")
    print(f"   mAP@50: {val_results.box.map50:.3f}")
    print(f"   mAP@50-95: {val_results.box.map:.3f}")

# ============================================================
# TEST PREDICTIONS
# ============================================================

print("\n" + "="*60)
print("RUNNING PREDICTIONS")
print("="*60 + "\n")

test_images_dir = dataset_dir / 'test' / 'images'
test_images = list(test_images_dir.glob('*.jpg'))[:10]

if test_images:
    print(f"Testing on {len(test_images)} sample images...\n")
    predictions = best_model.predict(
        source=test_images,
        conf=0.25,
        save=True,
        project='/kaggle/working',
        name='predictions',
        device=0
    )
    print(f"\n✅ Predictions saved to /kaggle/working/predictions")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "="*60)
print("🎉 ALL DONE!")
print("="*60)
print(f"\n📁 Results Location:")
print(f"   /kaggle/working/yolo11_footprint_detection/")
print(f"\n📊 Key Files:")
print(f"   - Best Model: weights/best.pt")
print(f"   - Last Model: weights/last.pt") 
print(f"   - Metrics: results.csv")
print(f"   - Plots: results.png, confusion_matrix.png")
print(f"\n💡 To use the model:")
print(f"   model = YOLO('/kaggle/working/yolo11_footprint_detection/weights/best.pt')")
print(f"   results = model.predict(source='image.jpg', conf=0.25)")
print("="*60)
```

---

## 🏃 Quick Run (If you already uploaded dataset_combined)

**Just paste this in a Kaggle notebook cell:**

```python
!pip install -q ultralytics
from ultralytics import YOLO

model = YOLO('yolo11m.pt')
model.train(
    data='/kaggle/input/dataset-combined/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    project='/kaggle/working'
)
```

---

## 📁 What You're Running

| File | Purpose |
|------|---------|
| `split_dataset.py` | ✅ Splits and combines your datasets |
| `train_yolo11_kaggle.py` | Trains locally (advanced) |
| `kaggle_yolo11_notebook.py` | Full notebook for Kaggle |
| `dataset_combined/` | ✅ Ready-to-use combined dataset |
| `data.yaml` | Dataset configuration (classes, paths) |

---

## 🎯 Expected Training Time

- **GPU (T4 on Kaggle)**: ~30-45 minutes for 50 epochs
- **CPU (Local)**: ~3-4 hours for 50 epochs

---

## ⚠️ Common Issues & Solutions

### Issue: "FileNotFoundError: data.yaml"
```
✅ Solution: Make sure dataset is uploaded correctly
   Path should be: /kaggle/input/[your-dataset-name]/data.yaml
```

### Issue: "CUDA out of memory"
```
✅ Solution: Reduce batch size
   Change: batch=16 → batch=8
```

### Issue: "GPU not detected"
```
✅ Solution: Enable GPU in Notebook Settings
   1. Click "Session" → "Accelerator" → "GPU"
   2. Select "T4"
   3. Run cells again
```

### Issue: "Module not found: ultralytics"
```
✅ Solution: Install in first cell
   !pip install -q ultralytics
```

---

## 📊 Model Sizes (Choose One)

| Size | Speed | mAP | Parameters | Size |
|------|-------|-----|------------|------|
| **nano** (n) | ⚡ Fastest | 73% | 2.6M | 6.2MB |
| **small** (s) | 🚀 Fast | 80% | 11.2M | 24MB |
| **medium** (m) | ⭐ Balanced | 83% | 25.9M | 51MB |
| **large** (l) | 🐢 Slow | 87% | 63.4M | 121MB |
| **xlarge** (x) | 🐢🐢 Slowest | 89% | 146.3M | 280MB |

**Recommended**: Use `yolo11m` (medium) for best balance

---

## 🚀 Advanced Options

### Train Longer
```python
epochs=100  # Instead of 50
```

### Different Model Size
```python
model = YOLO('yolo11l.pt')  # Use large model
```

### Lower Learning Rate
```python
lr0=0.005,  # Slower learning
```

### More Augmentation
```python
degrees=20,
translate=0.2,
scale=0.7,
```

---

## 📊 After Training - Use Your Model

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('/kaggle/working/yolo11_footprint_detection/weights/best.pt')

# Predict on image
results = model.predict(
    source='footprint.jpg',
    conf=0.25,
    save=True
)

# Get predictions
for r in results:
    for box in r.boxes:
        print(f"Class: {r.names[int(box.cls)]}")
        print(f"Confidence: {box.conf:.2f}")
```

---

## ✅ Verification Checklist

Before running on Kaggle:
- ✅ Dataset uploaded (dataset_combined folder)
- ✅ Notebook has GPU enabled
- ✅ data.yaml is in the dataset folder
- ✅ All images have corresponding labels
- ✅ Dependencies installed (`!pip install ultralytics`)

---

## 🎓 Next Steps

1. ✅ Upload `dataset_combined` to Kaggle
2. ✅ Create Kaggle notebook
3. ✅ Enable GPU (Session → Accelerator → GPU)
4. ✅ Copy & paste training code
5. ✅ Run all cells
6. ✅ Download `best.pt` model
7. ✅ Use model for inference

---

**Made with ❤️ - Good luck training!**
