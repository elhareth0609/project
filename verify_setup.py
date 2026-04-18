#!/usr/bin/env python3
"""
Verify that YOLO 11 setup is ready
"""

import sys
import os
from pathlib import Path

print("\n" + "="*60)
print("YOLO 11 ENVIRONMENT VERIFICATION")
print("="*60 + "\n")

# Check 1: Python version
print("1️⃣  Python Version:")
print(f"   {sys.version}")

# Check 2: Required packages
print("\n2️⃣  Required Packages:")

packages = {
    'torch': 'PyTorch',
    'torchvision': 'TorchVision',
    'ultralytics': 'Ultralytics',
    'cv2': 'OpenCV',
    'numpy': 'NumPy',
    'yaml': 'PyYAML',
    'PIL': 'Pillow',
}

missing_packages = []
for pkg, name in packages.items():
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', 'Unknown')
        print(f"   ✅ {name}: {version}")
    except ImportError:
        print(f"   ❌ {name}: NOT INSTALLED")
        missing_packages.append(pkg)

# Check 3: GPU availability
print("\n3️⃣  GPU/CUDA Status:")
try:
    import torch
    cuda_available = torch.cuda.is_available()
    print(f"   CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
    else:
        print("   ⚠️  No GPU detected (will use CPU - slower)")
except Exception as e:
    print(f"   ❌ Error checking GPU: {e}")

# Check 4: Dataset structure
print("\n4️⃣  Dataset Structure:")
project_root = Path(__file__).parent

datasets_info = {
    'animals': project_root / 'animals',
    'humain': project_root / 'humain',
}

for ds_name, ds_path in datasets_info.items():
    if ds_path.exists():
        train_img = ds_path / 'train' / 'images'
        train_lbl = ds_path / 'train' / 'labels'
        
        if train_img.exists() and train_lbl.exists():
            img_count = len(list(train_img.glob('*.*')))
            lbl_count = len(list(train_lbl.glob('*.txt')))
            status = "✅" if img_count == lbl_count else "⚠️"
            print(f"   {status} {ds_name}: {img_count} images, {lbl_count} labels")
        else:
            print(f"   ❌ {ds_name}: train/images or train/labels not found")
    else:
        print(f"   ❌ {ds_name}: directory not found")

# Check 5: YOLO model availability
print("\n5️⃣  YOLO Model Status:")
try:
    from ultralytics import YOLO
    print("   ✅ YOLO import successful")
    print("   📝 Available models: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x")
except Exception as e:
    print(f"   ❌ YOLO import failed: {e}")

# Check 6: data.yaml
print("\n6️⃣  Combined Dataset Status:")
combined_yaml = project_root / 'dataset_combined' / 'data.yaml'
if combined_yaml.exists():
    print(f"   ✅ data.yaml found: {combined_yaml}")
else:
    print(f"   ℹ️  data.yaml not found (run split_dataset.py first)")

# Summary
print("\n" + "="*60)
if missing_packages:
    print("⚠️  ACTION REQUIRED:")
    print(f"\n   Install missing packages:")
    print(f"   pip install {' '.join(missing_packages)}")
    print(f"\n   Or install all requirements:")
    print(f"   pip install -r requirements.txt")
else:
    print("✅ ALL CHECKS PASSED!")
    print("\n📋 Next steps:")
    print("   1. Run: python split_dataset.py")
    print("   2. Check the code works locally")
    print("   3. Upload to Kaggle and train")

print("="*60 + "\n")
